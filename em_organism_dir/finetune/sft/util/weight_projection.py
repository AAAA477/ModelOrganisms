"""
Weight Projection Callback for Emergent Misalignment Prevention

This module implements a proactive defense mechanism that projects out misalignment
directions from model weights after each optimizer step during fine-tuning.

The core idea: if misalignment has a predictable geometric signature in weight space
(a low-dimensional subspace), we can "vaccinate" the model by making those directions
off-limits during training.

Method: Weight Projection
    After each optimizer step, directly remove any component of the weights that
    aligns with the misalignment direction:

    W^(t) ← W^(t) - (W^(t) · u) u

    where u is the unit vector representing the misalignment direction.

LoRA-Space Projection:
    When training LoRA adapters, we must project in LoRA space, not full weight space.
    The LoRA parameters are:
        - lora_A: shape (rank, in_features)
        - lora_B: shape (out_features, rank)

    We extract the raw A and B matrices from "bad" LoRA adapters and use those
    as the directions to project out from the corresponding matrices during training.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class MisalignmentDirection:
    """
    Represents a misalignment direction vector for a specific layer/module.

    The direction is stored as a unit vector to ensure consistent projection
    regardless of the original magnitude.
    """

    def __init__(self, direction: torch.Tensor, layer_name: str):
        """
        Initialize a misalignment direction.

        Args:
            direction: The direction tensor (will be normalized to unit vector)
            layer_name: Name of the layer this direction applies to
        """
        self.layer_name = layer_name
        # Normalize to unit vector
        norm = torch.norm(direction.float())
        if norm > 0:
            self.direction = (direction.float() / norm)
        else:
            self.direction = direction.float()
        self._original_shape = direction.shape

    def to(self, device: torch.device, dtype: torch.dtype = None) -> "MisalignmentDirection":
        """Move direction to specified device and dtype."""
        self.direction = self.direction.to(device=device, dtype=dtype if dtype else self.direction.dtype)
        return self


class MisalignmentSubspace:
    """
    Represents a subspace of misalignment directions.

    When multiple misalignment directions are identified (e.g., from different
    harmful fine-tuning tasks), they can be combined into a subspace. The
    projection then removes components along all directions in the subspace.

    Directions are orthonormalized per layer via Gram-Schmidt when finalized,
    so projection is always the simple W - Uᵀ(UW) with no matrix inverse.
    """

    def __init__(self):
        self.directions: Dict[str, List[MisalignmentDirection]] = {}
        self._finalized = False

    def add_direction(self, direction: MisalignmentDirection):
        """Add a direction to the subspace for its layer."""
        if direction.layer_name not in self.directions:
            self.directions[direction.layer_name] = []
        self.directions[direction.layer_name].append(direction)
        self._finalized = False

    def orthonormalize(self):
        """
        Orthonormalize all directions per layer using Gram-Schmidt.

        After this, each layer's directions form an orthonormal basis.
        Linearly dependent directions are dropped. This ensures projection
        is always the simple W - Uᵀ(UW) with no solve needed.
        """
        for layer_name in self.directions:
            raw_tensors = [md.direction for md in self.directions[layer_name]]
            ortho_tensors = orthonormalize_directions(raw_tensors)
            self.directions[layer_name] = [
                MisalignmentDirection(t, layer_name) for t in ortho_tensors
            ]
        self._finalized = True

    def get_directions_for_layer(self, layer_name: str) -> List[MisalignmentDirection]:
        """Get all misalignment directions for a specific layer."""
        return self.directions.get(layer_name, [])

    def get_all_layer_names(self) -> List[str]:
        """Get names of all layers with registered directions."""
        return list(self.directions.keys())

    def to(self, device: torch.device, dtype: torch.dtype = None) -> "MisalignmentSubspace":
        """Move all directions to specified device and dtype."""
        for layer_name in self.directions:
            for direction in self.directions[layer_name]:
                direction.to(device, dtype)
        return self


def project_out_direction(
    weight: torch.Tensor,
    direction: torch.Tensor
) -> torch.Tensor:
    """
    Project out a direction from a weight tensor.

    Implements: W ← W - (W · u) u

    where u is the unit direction vector and · denotes the appropriate
    inner product (flattened dot product).

    Args:
        weight: The weight tensor to modify
        direction: The unit direction vector to project out (same shape as weight)

    Returns:
        The projected weight tensor
    """
    # Flatten both tensors for dot product
    weight_flat = weight.view(-1).float()
    direction_flat = direction.view(-1).float()

    # Compute projection coefficient: (W · u)
    projection_coeff = torch.dot(weight_flat, direction_flat)

    # Compute projection and subtract: W - (W · u) u
    projection = projection_coeff * direction_flat
    projected_weight = weight_flat - projection

    # Reshape back to original shape and dtype
    return projected_weight.view(weight.shape).to(weight.dtype)


def project_out_subspace(
    weight: torch.Tensor,
    directions: List[MisalignmentDirection]
) -> torch.Tensor:
    """
    Project out multiple directions (a subspace) from a weight tensor.

    For multiple directions, we apply sequential projection. For an orthonormal
    basis this is equivalent to projecting out the entire subspace at once.

    Args:
        weight: The weight tensor to modify
        directions: List of MisalignmentDirection objects to project out

    Returns:
        The projected weight tensor
    """
    result = weight
    for md in directions:
        result = project_out_direction(result, md.direction.view(weight.shape))
    return result


# =============================================================================
# Orthonormalization and Batch Projection
# =============================================================================

def orthonormalize_directions(
    directions: List[torch.Tensor],
    eps: float = 1e-8
) -> List[torch.Tensor]:
    """
    Orthonormalize a list of direction tensors using Gram-Schmidt.

    Takes raw (possibly non-orthogonal) direction vectors and produces an
    orthonormal basis spanning the same subspace. Directions that are linearly
    dependent on previous ones are dropped.

    Args:
        directions: List of direction tensors (same shape)
        eps: Minimum norm threshold — vectors below this are considered
             linearly dependent and dropped

    Returns:
        List of orthonormal direction tensors (may be shorter than input
        if some directions were linearly dependent)
    """
    if len(directions) == 0:
        return []

    shape = directions[0].shape
    basis = []

    for d in directions:
        v = d.view(-1).float()

        # Subtract projections onto existing basis vectors
        for b in basis:
            v = v - torch.dot(v, b) * b

        # Normalize
        norm = torch.norm(v)
        if norm > eps:
            basis.append(v / norm)

    # Reshape back to original shape
    return [b.view(shape) for b in basis]


def axial_align(vectors: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Sign-align vectors to a common mean axis.

    This mirrors the notebook logic: normalize each vector, compute a centroid
    axis, flip vectors with negative dot product, then recompute the centroid.
    """
    if not vectors:
        return [], None

    unit_vectors = []
    for v in vectors:
        flat = v.view(-1).float()
        norm = torch.linalg.norm(flat)
        unit_vectors.append(flat if norm <= 1e-12 else flat / norm)

    centroid = torch.stack(unit_vectors, dim=0).mean(dim=0)
    centroid_norm = torch.linalg.norm(centroid)
    if centroid_norm > 1e-12:
        centroid = centroid / centroid_norm

    aligned = [u if torch.dot(u, centroid) >= 0 else -u for u in unit_vectors]
    centroid = torch.stack(aligned, dim=0).mean(dim=0)
    centroid_norm = torch.linalg.norm(centroid)
    if centroid_norm > 1e-12:
        centroid = centroid / centroid_norm

    return aligned, centroid


def qr_row_basis(
    vectors: List[torch.Tensor],
    tol: float = 1e-7,
    max_rank: Optional[int] = None,
) -> torch.Tensor:
    """Build an orthonormal row basis U (k, d) using thin QR on V^T."""
    if not vectors:
        return torch.empty(0, 0, dtype=torch.float32)

    V = torch.stack([v.view(-1).float() for v in vectors], dim=0)  # (n, d)
    Q, R = torch.linalg.qr(V.T, mode="reduced")
    diag = torch.diagonal(R).abs()
    rank = int((diag > tol).sum().item())
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return Q[:, :rank].T.contiguous().to(dtype=torch.float32)


def svd_row_basis(
    vectors: List[torch.Tensor],
    max_rank: Optional[int] = None,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Build an orthonormal row basis U (k, d) via uncentered SVD on stacked vectors."""
    if not vectors:
        return torch.empty(0, 0, dtype=torch.float32)

    X = torch.stack([v.view(-1).float() for v in vectors], dim=0)  # (n, d)
    _, singular_values, vh = torch.linalg.svd(X, full_matrices=False)
    keep = singular_values > tol
    basis = vh[keep]
    if max_rank is not None:
        basis = basis[: int(max_rank)]
    return basis.contiguous().to(dtype=torch.float32)


_LORA_SITE_RE = re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.([^.]+)$")


def parse_lora_site(base_name: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract layer index and site type from a LoRA base name."""
    match = _LORA_SITE_RE.search(base_name)
    if not match:
        return None, None
    layer_idx = int(match.group(1))
    site_type = f"{match.group(2)}.{match.group(3)}"
    return layer_idx, site_type


def batch_project_out_subspace(
    weight: torch.Tensor,
    directions: List[MisalignmentDirection],
    _cached_basis: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Project out an entire subspace in one batched matrix operation.

    Directions are assumed to be orthonormal (enforced at subspace construction
    time via Gram-Schmidt). The projection is:

        W_proj = W - Uᵀ(UW)

    where U is (k, d) with each row an orthonormal direction vector.

    Args:
        weight: The weight tensor to modify
        directions: List of MisalignmentDirection objects to project out
        _cached_basis: Optional pre-stacked basis matrix (k, d) to skip re-stacking

    Returns:
        The projected weight tensor
    """
    if len(directions) == 0:
        return weight

    # Single direction: fast path
    if len(directions) == 1:
        return project_out_direction(weight, directions[0].direction.view(weight.shape))

    original_shape = weight.shape
    original_dtype = weight.dtype
    weight_flat = weight.view(-1).float()

    # Stack directions into matrix U: (k, d)
    if _cached_basis is not None:
        U = _cached_basis
    else:
        U = torch.stack(
            [md.direction.view(-1).float() for md in directions], dim=0
        )  # (k, d)

    # Orthonormal projection: W - Uᵀ(UW)
    coeffs = U @ weight_flat  # (k,)
    projection = U.t() @ coeffs  # (d,)

    projected = weight_flat - projection
    return projected.view(original_shape).to(original_dtype)


def _get_projection_basis(
    tensor: torch.Tensor,
    directions: List[MisalignmentDirection],
    cached_basis: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Return a stacked orthonormal basis matching the tensor device."""
    if len(directions) == 0:
        return None
    if cached_basis is not None:
        return cached_basis.to(device=tensor.device)
    return torch.stack(
        [md.direction.view(-1).float().to(tensor.device) for md in directions],
        dim=0,
    )


def _subspace_component_norm(
    tensor: torch.Tensor,
    directions: List[MisalignmentDirection],
    cached_basis: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Return total tensor norm and norm of the component inside the blocked subspace."""
    if len(directions) == 0:
        return 0.0, 0.0

    tensor_flat = tensor.detach().view(-1).float()
    total_norm = torch.norm(tensor_flat).item()
    if total_norm == 0.0:
        return 0.0, 0.0

    basis = _get_projection_basis(tensor, directions, cached_basis)
    if basis is None:
        return total_norm, 0.0

    coeffs = basis @ tensor_flat
    blocked_norm = torch.norm(coeffs).item()
    return total_norm, blocked_norm


def _format_ratio(numerator: float, denominator: float) -> str:
    if denominator == 0.0:
        return "0.00%"
    return f"{(100.0 * numerator / denominator):.2f}%"


def _basis_component_norm(tensor: torch.Tensor, basis: torch.Tensor) -> Tuple[float, float]:
    """Return total norm and basis-aligned norm for shared-basis projection tensors."""
    total_norm = torch.norm(tensor.detach().float()).item()
    if total_norm == 0.0:
        return 0.0, 0.0

    coeffs = basis.float().to(tensor.device) @ tensor.detach().float()
    blocked_norm = torch.norm(coeffs).item()
    return total_norm, blocked_norm


class WeightProjectionCallback(TrainerCallback):
    """
    HuggingFace Trainer callback that projects out misalignment directions
    from model weights after each optimizer step.

    This implements the "Weight Projection" defense mechanism:
    - After every optimizer step, the callback checks all trainable parameters
    - For each parameter that has a registered misalignment direction, it
      projects out the component along that direction
    - This prevents the model from moving toward known misalignment regions
      in weight space

    Usage:
        ```python
        # Load or compute misalignment directions
        subspace = MisalignmentSubspace()
        subspace.add_direction(MisalignmentDirection(direction_tensor, "layer_name"))

        # Create callback
        callback = WeightProjectionCallback(
            misalignment_subspace=subspace,
            projection_strength=1.0,  # Full projection
            apply_every_n_steps=1,    # Every step
        )

        # Add to trainer
        trainer = Trainer(..., callbacks=[callback])
        ```
    """

    def __init__(
        self,
        misalignment_subspace: MisalignmentSubspace,
        projection_strength: float = 1.0,
        apply_every_n_steps: int = 1,
        target_modules: Optional[List[str]] = None,
        verbose: bool = False,
        debug: bool = False,
        debug_every_n_steps: int = 1,
        debug_max_layers: int = 8,
    ):
        """
        Initialize the weight projection callback.

        Args:
            misalignment_subspace: The subspace of misalignment directions to project out
            projection_strength: How much of the projection to apply (0.0 to 1.0).
                                1.0 means full projection, 0.5 means half, etc.
                                This can be useful for gradual/soft projection.
            apply_every_n_steps: Apply projection every N optimizer steps.
                                Default is 1 (every step).
            target_modules: Optional list of module name patterns to apply projection to.
                           If None, applies to all modules with registered directions.
            verbose: If True, print projection statistics during training.
            debug: If True, print per-layer projection diagnostics.
            debug_every_n_steps: Print diagnostics every N optimizer steps.
            debug_max_layers: Maximum number of layers to print per diagnostic step.
        """
        self.misalignment_subspace = misalignment_subspace
        self.projection_strength = projection_strength
        self.apply_every_n_steps = apply_every_n_steps
        self.target_modules = target_modules
        self.verbose = verbose
        self.debug = debug
        self.debug_every_n_steps = debug_every_n_steps
        self.debug_max_layers = debug_max_layers
        self._step_count = 0
        self._initialized = False
        self._layer_to_param: Dict[str, torch.nn.Parameter] = {}
        self._debug_grad_stats: Dict[str, Dict[str, float]] = {}

    def _initialize_layer_mapping(self, model: torch.nn.Module):
        """
        Build mapping from layer names in the subspace to actual model parameters.

        This handles various naming conventions:
        - Full parameter names (e.g., "model.layers.0.mlp.down_proj.weight")
        - LoRA adapter names (e.g., "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight")
        """
        self._layer_to_param = {}
        subspace_layers = set(self.misalignment_subspace.get_all_layer_names())

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this parameter matches any subspace layer
            for layer_name in subspace_layers:
                if self._matches_layer(name, layer_name):
                    self._layer_to_param[layer_name] = param
                    if self.verbose:
                        print(f"[WeightProjection] Mapped {layer_name} -> {name}")
                    break

        # Move subspace directions to the right device/dtype
        if len(self._layer_to_param) > 0:
            sample_param = next(iter(self._layer_to_param.values()))
            self.misalignment_subspace.to(sample_param.device, sample_param.dtype)

        self._initialized = True

        if self.verbose:
            print(f"[WeightProjection] Initialized with {len(self._layer_to_param)} layers mapped")

    def _matches_layer(self, param_name: str, layer_name: str) -> bool:
        """
        Check if a parameter name matches a layer name from the subspace.

        Handles various naming conventions and module patterns.
        """
        # Direct match
        if layer_name in param_name:
            return True

        # Check target_modules filter if specified
        if self.target_modules:
            if not any(module in param_name for module in self.target_modules):
                return False

        # Normalize and compare
        # Remove common prefixes
        normalized_param = param_name.replace("base_model.model.", "").replace("model.", "")
        normalized_layer = layer_name.replace("base_model.model.", "").replace("model.", "")

        return normalized_layer in normalized_param

    def _apply_projection(self, model: torch.nn.Module):
        """Apply weight projection to all mapped parameters."""
        total_projection_norm = 0.0
        num_projected = 0
        debug_rows = []

        for layer_name, param in self._layer_to_param.items():
            directions = self.misalignment_subspace.get_directions_for_layer(layer_name)
            if not directions:
                continue

            with torch.no_grad():
                original_weight = param.data.clone()
                projected_weight = project_out_subspace(param.data, directions)

                # Apply projection strength (interpolate between original and projected)
                if self.projection_strength < 1.0:
                    param.data = (
                        self.projection_strength * projected_weight +
                        (1.0 - self.projection_strength) * original_weight
                    )
                else:
                    param.data = projected_weight

                # Track statistics
                diff = original_weight - param.data
                total_projection_norm += torch.norm(diff).item()
                num_projected += 1

                if self.debug and self._step_count % self.debug_every_n_steps == 0:
                    weight_norm_before, blocked_before = _subspace_component_norm(
                        original_weight, directions
                    )
                    weight_norm_after, blocked_after = _subspace_component_norm(
                        param.data, directions
                    )
                    grad_stats = self._debug_grad_stats.get(layer_name, {})
                    debug_rows.append(
                        {
                            "layer_name": layer_name,
                            "grad_norm": grad_stats.get("grad_norm", 0.0),
                            "grad_blocked_norm": grad_stats.get("grad_blocked_norm", 0.0),
                            "weight_norm_before": weight_norm_before,
                            "blocked_before": blocked_before,
                            "weight_norm_after": weight_norm_after,
                            "blocked_after": blocked_after,
                            "removed_norm": torch.norm(diff).item(),
                        }
                    )

        if self.verbose and num_projected > 0:
            avg_norm = total_projection_norm / num_projected
            print(f"[WeightProjection] Step {self._step_count}: "
                  f"Projected {num_projected} layers, avg norm change: {avg_norm:.6f}")
        if debug_rows:
            self._print_debug_rows("WeightProjection", debug_rows)

    def _print_debug_rows(self, prefix: str, rows: List[Dict[str, float]]):
        rows = sorted(rows, key=lambda row: row["blocked_before"], reverse=True)
        print(f"[{prefix}][Debug] Step {self._step_count}: showing top {min(len(rows), self.debug_max_layers)} layers")
        for row in rows[:self.debug_max_layers]:
            print(
                f"[{prefix}][Debug] {row['layer_name']}: "
                f"grad blocked {row['grad_blocked_norm']:.6f}/{row['grad_norm']:.6f} "
                f"({_format_ratio(row['grad_blocked_norm'], row['grad_norm'])}), "
                f"weight blocked {row['blocked_before']:.6f}/{row['weight_norm_before']:.6f} -> "
                f"{row['blocked_after']:.6f}/{row['weight_norm_after']:.6f} "
                f"(removed {row['removed_norm']:.6f})"
            )

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs
    ):
        """Capture gradient overlap with blocked directions before the optimizer step."""
        if not self.debug or model is None:
            return

        if not self._initialized:
            self._initialize_layer_mapping(model)

        upcoming_step = self._step_count + 1
        if upcoming_step % self.apply_every_n_steps != 0:
            return
        if upcoming_step % self.debug_every_n_steps != 0:
            return

        self._debug_grad_stats = {}
        for layer_name, param in self._layer_to_param.items():
            if param.grad is None:
                continue
            directions = self.misalignment_subspace.get_directions_for_layer(layer_name)
            if not directions:
                continue
            grad_norm, grad_blocked_norm = _subspace_component_norm(param.grad, directions)
            self._debug_grad_stats[layer_name] = {
                "grad_norm": grad_norm,
                "grad_blocked_norm": grad_blocked_norm,
            }

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs
    ):
        """
        Called after each optimizer step.

        This is where we apply the weight projection to remove misalignment
        components from the updated weights.
        """
        self._step_count += 1

        if model is None:
            return

        # Initialize layer mapping on first step
        if not self._initialized:
            self._initialize_layer_mapping(model)

        # Check if we should apply projection this step
        if self._step_count % self.apply_every_n_steps != 0:
            return

        # Apply projection
        self._apply_projection(model)


def load_misalignment_directions_from_file(
    filepath: Union[str, Path],
    device: torch.device = None
) -> MisalignmentSubspace:
    """
    Load misalignment directions from a saved file.

    Expected file format (PyTorch .pt file):
    {
        "layer_name_1": tensor_of_shape_matching_layer,
        "layer_name_2": tensor_of_shape_matching_layer,
        ...
    }

    Args:
        filepath: Path to the .pt file containing direction tensors
        device: Device to load tensors to (default: CPU)

    Returns:
        MisalignmentSubspace populated with the loaded directions
    """
    if device is None:
        device = torch.device("cpu")

    data = torch.load(filepath, map_location=device)
    subspace = MisalignmentSubspace()

    for layer_name, direction_tensor in data.items():
        direction = MisalignmentDirection(direction_tensor, layer_name)
        subspace.add_direction(direction)

    subspace.orthonormalize()
    return subspace


def compute_misalignment_direction_from_lora(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    alpha: float,
    rank: int
) -> torch.Tensor:
    """
    Compute the effective weight change direction from LoRA matrices.

    LoRA represents weight changes as: ΔW = (α/r) * B @ A

    This function computes the normalized direction of this weight change,
    which can be used as a misalignment direction if the LoRA was trained
    on a misalignment-inducing task.

    Args:
        lora_A: LoRA A matrix of shape (rank, in_features)
        lora_B: LoRA B matrix of shape (out_features, rank)
        alpha: LoRA alpha scaling parameter
        rank: LoRA rank

    Returns:
        Normalized direction tensor of shape (out_features, in_features)
    """
    # Compute effective weight change
    scale = alpha / rank
    delta_w = scale * (lora_B @ lora_A)

    # Normalize to unit vector
    norm = torch.norm(delta_w)
    if norm > 0:
        delta_w = delta_w / norm

    return delta_w


def create_subspace_from_lora_models(
    lora_repo_ids: List[str],
    target_layers: Optional[List[str]] = None,
    quiet: bool = True,
    average_directions: bool = False
) -> MisalignmentSubspace:
    """
    Create a misalignment subspace from multiple LoRA models.

    This is useful when you have multiple LoRA models trained on different
    harmful tasks and want to protect against all of them.

    Args:
        lora_repo_ids: List of HuggingFace repo IDs containing LoRA models
        target_layers: Optional list of layer patterns to include.
                      If None, includes all available layers.
        quiet: If True, suppress download progress bars
        average_directions: If True, average the directions from all LoRA models
                           into a single direction per layer. If False, keep all
                           directions separately in the subspace.

    Returns:
        MisalignmentSubspace containing directions from all LoRA models
    """
    from em_organism_dir.lora_interp.lora_utils import (
        get_lora_components_per_layer,
        get_layer_number
    )

    if average_directions:
        return _create_averaged_subspace_from_lora_models(
            lora_repo_ids, target_layers, quiet
        )

    subspace = MisalignmentSubspace()

    for repo_id in lora_repo_ids:
        try:
            components = get_lora_components_per_layer(repo_id, quiet=quiet)

            for layer_name, lora_comp in components.components.items():
                # Filter by target layers if specified
                if target_layers:
                    if not any(t in layer_name for t in target_layers):
                        continue

                # Compute direction from LoRA matrices
                direction_tensor = compute_misalignment_direction_from_lora(
                    lora_A=lora_comp.A,
                    lora_B=lora_comp.B,
                    alpha=lora_comp.alpha.item() if hasattr(lora_comp.alpha, 'item') else lora_comp.alpha,
                    rank=lora_comp.A.shape[0]
                )

                direction = MisalignmentDirection(direction_tensor, layer_name)
                subspace.add_direction(direction)

        except Exception as e:
            print(f"Warning: Failed to load LoRA from {repo_id}: {e}")

    subspace.orthonormalize()
    return subspace


def _create_averaged_subspace_from_lora_models(
    lora_repo_ids: List[str],
    target_layers: Optional[List[str]] = None,
    quiet: bool = True
) -> MisalignmentSubspace:
    """
    Create a misalignment subspace with averaged directions from multiple LoRA models.

    For each layer, this computes the average of the misalignment directions from
    all LoRA models and uses that as the single direction to project out.

    This implements the formula:
        u_avg = normalize(sum(u_i) / N)

    where u_i are the unit directions from each LoRA model.

    Args:
        lora_repo_ids: List of HuggingFace repo IDs containing LoRA models
        target_layers: Optional list of layer patterns to include.
        quiet: If True, suppress download progress bars

    Returns:
        MisalignmentSubspace with one averaged direction per layer
    """
    from em_organism_dir.lora_interp.lora_utils import get_lora_components_per_layer

    # Collect all directions per layer
    layer_directions: Dict[str, List[torch.Tensor]] = {}

    for repo_id in lora_repo_ids:
        try:
            print(f"[WeightProjection] Loading LoRA from: {repo_id}")
            components = get_lora_components_per_layer(repo_id, quiet=quiet)

            for layer_name, lora_comp in components.components.items():
                # Filter by target layers if specified
                if target_layers:
                    if not any(t in layer_name for t in target_layers):
                        continue

                # Compute direction from LoRA matrices
                direction_tensor = compute_misalignment_direction_from_lora(
                    lora_A=lora_comp.A,
                    lora_B=lora_comp.B,
                    alpha=lora_comp.alpha.item() if hasattr(lora_comp.alpha, 'item') else lora_comp.alpha,
                    rank=lora_comp.A.shape[0]
                )

                if layer_name not in layer_directions:
                    layer_directions[layer_name] = []
                layer_directions[layer_name].append(direction_tensor)

        except Exception as e:
            print(f"Warning: Failed to load LoRA from {repo_id}: {e}")

    # Compute averaged directions and create subspace
    subspace = MisalignmentSubspace()

    for layer_name, directions in layer_directions.items():
        if len(directions) == 0:
            continue

        # Stack and average the directions
        stacked = torch.stack(directions, dim=0)  # [N, out_features, in_features]
        averaged = stacked.mean(dim=0)  # [out_features, in_features]

        # Create normalized direction
        direction = MisalignmentDirection(averaged, layer_name)
        subspace.add_direction(direction)

        print(f"[WeightProjection] Layer {layer_name}: averaged {len(directions)} directions")

    subspace.orthonormalize()
    return subspace


def compute_and_save_averaged_directions(
    lora_repo_ids: List[str],
    output_path: Union[str, Path],
    target_layers: Optional[List[str]] = None,
    quiet: bool = True
) -> None:
    """
    Compute averaged misalignment directions from multiple LoRA models and save to file.

    This is useful for pre-computing directions once and reusing them across
    multiple training runs.

    Args:
        lora_repo_ids: List of HuggingFace repo IDs containing LoRA models
        output_path: Path to save the directions (.pt file)
        target_layers: Optional list of layer patterns to include.
        quiet: If True, suppress download progress bars

    Example:
        >>> compute_and_save_averaged_directions(
        ...     lora_repo_ids=[
        ...         "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
        ...         "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train",
        ...         "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_sports_extended_train",
        ...     ],
        ...     output_path="misalignment_directions_averaged.pt"
        ... )
    """
    subspace = _create_averaged_subspace_from_lora_models(
        lora_repo_ids, target_layers, quiet
    )

    # Convert to saveable format
    directions_dict = {}
    for layer_name in subspace.get_all_layer_names():
        directions = subspace.get_directions_for_layer(layer_name)
        if directions:
            # Take the first (and should be only) direction for averaged subspace
            directions_dict[layer_name] = directions[0].direction

    torch.save(directions_dict, output_path)
    print(f"[WeightProjection] Saved averaged directions for {len(directions_dict)} layers to {output_path}")


# =============================================================================
# LoRA-Space Projection (for training LoRA adapters)
# =============================================================================

def extract_all_lora_components_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    config,
    target_modules: Optional[List[str]] = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract ALL LoRA A and B matrices from a state dict.

    Unlike extract_mlp_downproj_components which only extracts mlp.down_proj,
    this extracts all LoRA components matching target_modules.

    Args:
        state_dict: The loaded LoRA state dict
        config: PeftConfig for the model
        target_modules: Optional list of module patterns to filter (e.g., ["down_proj", "q_proj"])
                       If None, extracts all available LoRA components.

    Returns:
        Dict mapping layer base names to {"A": tensor, "B": tensor, "alpha": float}
    """
    components = {}

    # Find all unique base layer names
    for key in state_dict:
        if ".lora_A.weight" in key:
            base = key.replace(".lora_A.weight", "")

            # Filter by target modules if specified
            if target_modules:
                if not any(module in base for module in target_modules):
                    continue

            A = state_dict.get(f"{base}.lora_A.weight")
            B = state_dict.get(f"{base}.lora_B.weight")

            if A is None or B is None:
                continue

            # Get alpha (usually global from config)
            alpha_tensor = state_dict.get(f"{base}.alpha")
            if alpha_tensor is not None:
                alpha = alpha_tensor.item()
            else:
                config_dict = config.to_dict()
                alpha = float(config_dict.get("lora_alpha", 16))

            components[base] = {
                "A": A,
                "B": B,
                "alpha": alpha,
                "rank": A.shape[0]
            }

    return components


def create_lora_subspace_from_lora_models(
    lora_repo_ids: List[str],
    target_modules: Optional[List[str]] = None,
    quiet: bool = True,
    average_directions: bool = True
) -> MisalignmentSubspace:
    """
    Create a misalignment subspace that operates in LoRA space (on A and B matrices directly).

    This is the correct approach when training LoRA adapters, as it projects out
    directions from the actual trainable parameters (lora_A and lora_B) rather than
    from the composed weight change (B @ A).

    Args:
        lora_repo_ids: List of HuggingFace repo IDs containing bad LoRA models
        target_modules: Optional list of module patterns to include
        quiet: If True, suppress download progress bars
        average_directions: If True, average directions across all LoRA models

    Returns:
        MisalignmentSubspace with directions for both lora_A and lora_B parameters
    """
    from em_organism_dir.lora_interp.lora_utils import download_lora_weights, load_lora_state_dict

    # Collect A and B matrices separately for each layer
    layer_A_directions: Dict[str, List[torch.Tensor]] = {}
    layer_B_directions: Dict[str, List[torch.Tensor]] = {}

    for repo_id in lora_repo_ids:
        try:
            print(f"[LoRAProjection] Loading LoRA from: {repo_id}")
            lora_path, config = download_lora_weights(repo_id, quiet=quiet)
            state_dict = load_lora_state_dict(lora_path)

            components = extract_all_lora_components_from_state_dict(
                state_dict, config, target_modules
            )

            for base_name, comp in components.items():
                # Create keys for A and B matrices
                # The base_name is like "base_model.model.model.layers.0.mlp.down_proj"
                # We create separate keys for A and B
                a_key = f"{base_name}.lora_A"
                b_key = f"{base_name}.lora_B"

                if a_key not in layer_A_directions:
                    layer_A_directions[a_key] = []
                if b_key not in layer_B_directions:
                    layer_B_directions[b_key] = []

                # Store the raw A and B matrices (will be normalized later)
                layer_A_directions[a_key].append(comp["A"].clone())
                layer_B_directions[b_key].append(comp["B"].clone())

        except Exception as e:
            print(f"Warning: Failed to load LoRA from {repo_id}: {e}")

    # Create subspace with averaged or individual directions
    subspace = MisalignmentSubspace()

    # Process A matrices
    for layer_name, matrices in layer_A_directions.items():
        if len(matrices) == 0:
            continue

        if average_directions:
            # Average and normalize
            stacked = torch.stack(matrices, dim=0)
            averaged = stacked.mean(dim=0)
            direction = MisalignmentDirection(averaged, layer_name)
            subspace.add_direction(direction)
            print(f"[LoRAProjection] {layer_name}: averaged {len(matrices)} A matrices, shape {averaged.shape}")
        else:
            for mat in matrices:
                direction = MisalignmentDirection(mat, layer_name)
                subspace.add_direction(direction)

    # Process B matrices
    for layer_name, matrices in layer_B_directions.items():
        if len(matrices) == 0:
            continue

        if average_directions:
            stacked = torch.stack(matrices, dim=0)
            averaged = stacked.mean(dim=0)
            direction = MisalignmentDirection(averaged, layer_name)
            subspace.add_direction(direction)
            print(f"[LoRAProjection] {layer_name}: averaged {len(matrices)} B matrices, shape {averaged.shape}")
        else:
            for mat in matrices:
                direction = MisalignmentDirection(mat, layer_name)
                subspace.add_direction(direction)

    subspace.orthonormalize()
    print(f"[LoRAProjection] Created subspace with {len(subspace.get_all_layer_names())} layer directions")
    return subspace


def create_module_shared_basis_from_lora_models(
    lora_repo_ids: List[str],
    target_modules: Optional[List[str]] = None,
    quiet: bool = True,
    target_rank: int = 32,
    qr_tol: float = 1e-7,
    svd_tol: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    """
    Build a module-wise shared orthonormal basis from LoRA B matrices.

    This ports the notebook workflow into LoRA B-space:
    1. For each model and module type, collect B columns as vectors in output space
    2. Sign-align those vectors and build a per-model QR basis
    3. Pool the QR bases across models and run SVD to get a shared basis

    Returns:
        Dict like {"self_attn.q_proj": basis, ...} where each basis is (k, hidden_size)
    """
    from em_organism_dir.lora_interp.lora_utils import download_lora_weights, load_lora_state_dict

    site_qr_bases: Dict[str, List[torch.Tensor]] = {}

    for repo_id in lora_repo_ids:
        try:
            print(f"[SharedBasisProjection] Loading LoRA from: {repo_id}")
            lora_path, config = download_lora_weights(repo_id, quiet=quiet)
            state_dict = load_lora_state_dict(lora_path)
            components = extract_all_lora_components_from_state_dict(state_dict, config, target_modules)

            per_site_vectors: Dict[str, List[torch.Tensor]] = {}
            for base_name, comp in components.items():
                _, site_type = parse_lora_site(base_name)
                if site_type is None:
                    continue

                B = comp["B"].float()
                for col_idx in range(B.shape[1]):
                    per_site_vectors.setdefault(site_type, []).append(B[:, col_idx].clone())

            for site_type, vectors in per_site_vectors.items():
                aligned, _ = axial_align(vectors)
                basis = qr_row_basis(aligned, tol=qr_tol, max_rank=target_rank)
                if basis.numel() > 0:
                    site_qr_bases.setdefault(site_type, []).append(basis.cpu())
                    print(
                        f"[SharedBasisProjection] {repo_id} {site_type}: "
                        f"QR basis shape {tuple(basis.shape)}"
                    )

        except Exception as e:
            print(f"Warning: Failed to load LoRA from {repo_id}: {e}")

    shared_bases: Dict[str, torch.Tensor] = {}
    for site_type, bases in site_qr_bases.items():
        pooled_vectors = [row for basis in bases for row in basis]
        shared_basis = svd_row_basis(
            pooled_vectors, max_rank=target_rank, tol=svd_tol
        )
        if shared_basis.numel() > 0:
            shared_bases[site_type] = shared_basis.cpu()
            print(
                f"[SharedBasisProjection] {site_type}: pooled {len(pooled_vectors)} vectors "
                f"-> shared basis shape {tuple(shared_basis.shape)}"
            )

    print(f"[SharedBasisProjection] Created shared module basis for {len(shared_bases)} module types")
    return shared_bases


class LoRAProjectionCallback(TrainerCallback):
    """
    HuggingFace Trainer callback that projects out misalignment directions
    from LoRA adapter weights (A and B matrices) after each optimizer step.

    This is the correct approach when training LoRA adapters, as it operates
    directly on the trainable parameters rather than the composed weight change.

    The key difference from WeightProjectionCallback:
    - WeightProjectionCallback: projects directions from ΔW = B @ A (wrong shapes for LoRA training)
    - LoRAProjectionCallback: projects directions from raw A and B matrices (correct shapes)
    """

    def __init__(
        self,
        misalignment_subspace: MisalignmentSubspace,
        projection_strength: float = 1.0,
        apply_every_n_steps: int = 1,
        verbose: bool = False,
        project_optimizer_states: bool = True,
        project_gradients: bool = False,
        gradient_projection_strength: float = 1.0,
        project_second_moment: bool = False,
        debug: bool = False,
        debug_every_n_steps: int = 1,
        debug_max_layers: int = 8,
    ):
        """
        Initialize the LoRA projection callback.

        Args:
            misalignment_subspace: Subspace with directions for lora_A and lora_B parameters
            projection_strength: How much of the projection to apply (0.0 to 1.0)
            apply_every_n_steps: Apply projection every N optimizer steps
            verbose: If True, print projection statistics
            project_optimizer_states: If True, also project out misalignment components
                from Adam's first moment (exp_avg / momentum). This prevents the
                optimizer from "remembering" the misalignment direction and pushing
                back harder each step.
            project_gradients: If True, project gradients before the optimizer step
                so forbidden components never reach Adam in the first place.
            gradient_projection_strength: How much of the gradient projection to
                apply (0.0 to 1.0).
            project_second_moment: If True, also project Adam's second-moment
                estimate (exp_avg_sq). This is a heuristic rather than an exact
                projection because squared-gradient statistics are not linear.
            debug: If True, print per-layer gradient and weight projection diagnostics.
            debug_every_n_steps: Print diagnostics every N optimizer steps.
            debug_max_layers: Maximum number of layers to print per diagnostic step.
        """
        self.misalignment_subspace = misalignment_subspace
        self.projection_strength = projection_strength
        self.apply_every_n_steps = apply_every_n_steps
        self.verbose = verbose
        self.project_optimizer_states = project_optimizer_states
        self.project_gradients = project_gradients
        self.gradient_projection_strength = gradient_projection_strength
        self.project_second_moment = project_second_moment
        self.debug = debug
        self.debug_every_n_steps = debug_every_n_steps
        self.debug_max_layers = debug_max_layers
        self._step_count = 0
        self._initialized = False
        self._layer_to_param: Dict[str, torch.nn.Parameter] = {}
        # Cache stacked basis matrices per layer to avoid re-stacking each step
        self._cached_bases: Dict[str, torch.Tensor] = {}
        # Map param -> optimizer state group for optimizer state projection
        self._param_to_optimizer_state: Dict[torch.nn.Parameter, dict] = {}
        self._debug_grad_stats: Dict[str, Dict[str, float]] = {}

    def _initialize_layer_mapping(self, model: torch.nn.Module):
        """Build mapping from subspace layer names to actual model parameters."""
        self._layer_to_param = {}
        subspace_layers = set(self.misalignment_subspace.get_all_layer_names())

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Match parameter names to subspace layer names
            # Parameter name: "base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight"
            # Subspace key:   "base_model.model.model.layers.0.mlp.down_proj.lora_A"
            for layer_name in subspace_layers:
                if layer_name in name and name.endswith(".weight"):
                    # Verify shapes match
                    directions = self.misalignment_subspace.get_directions_for_layer(layer_name)
                    if directions and directions[0].direction.shape == param.shape:
                        self._layer_to_param[layer_name] = param
                        if self.verbose:
                            print(f"[LoRAProjection] Mapped {layer_name} -> {name} (shape {param.shape})")
                        break
                    elif directions and self.verbose:
                        print(f"[LoRAProjection] Shape mismatch for {layer_name}: "
                              f"direction {directions[0].direction.shape} vs param {param.shape}")

        # Move subspace directions to the right device/dtype and pre-cache stacked bases
        if len(self._layer_to_param) > 0:
            sample_param = next(iter(self._layer_to_param.values()))
            self.misalignment_subspace.to(sample_param.device, sample_param.dtype)

            # Pre-stack direction matrices for batch projection (avoids per-step allocation)
            for layer_name in self._layer_to_param:
                directions = self.misalignment_subspace.get_directions_for_layer(layer_name)
                if directions and len(directions) > 1:
                    U = torch.stack(
                        [md.direction.view(-1).float().to(sample_param.device)
                         for md in directions], dim=0
                    )  # (k, d)
                    self._cached_bases[layer_name] = U

        self._initialized = True
        print(f"[LoRAProjection] Initialized with {len(self._layer_to_param)} layers mapped")

    def _project_single_tensor(
        self, tensor: torch.Tensor, directions: List[MisalignmentDirection],
        cached_basis: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Project out misalignment directions from a single tensor (weight or optimizer state)."""
        return batch_project_out_subspace(tensor, directions, _cached_basis=cached_basis)

    def _project_gradients(self):
        """Project gradients before the optimizer step so forbidden components are never applied."""
        if not self.project_gradients or self.gradient_projection_strength <= 0.0:
            return

        total_projection_norm = 0.0
        num_projected = 0

        for layer_name, param in self._layer_to_param.items():
            if param.grad is None:
                continue

            directions = self.misalignment_subspace.get_directions_for_layer(layer_name)
            if not directions:
                continue

            cached = self._cached_bases.get(layer_name)
            with torch.no_grad():
                original_grad = param.grad.data.clone()
                projected_grad = self._project_single_tensor(
                    param.grad.data, directions, cached
                )

                if self.gradient_projection_strength < 1.0:
                    param.grad.data = (
                        self.gradient_projection_strength * projected_grad +
                        (1.0 - self.gradient_projection_strength) * original_grad
                    )
                else:
                    param.grad.data = projected_grad

                total_projection_norm += torch.norm(original_grad - param.grad.data).item()
                num_projected += 1

        if self.verbose and num_projected > 0:
            avg_norm = total_projection_norm / num_projected
            print(f"[LoRAProjection] Step {self._step_count + 1}: "
                  f"Projected gradients for {num_projected} layers, avg norm change: {avg_norm:.6f}")

    def _project_optimizer_state_for_param(
        self, param: torch.nn.Parameter, directions: List[MisalignmentDirection],
        cached_basis: Optional[torch.Tensor], optimizer
    ):
        """
        Project out misalignment directions from Adam's internal state for a parameter.

        Adam maintains per-parameter state:
          - exp_avg (first moment / momentum): running mean of gradients
          - exp_avg_sq (second moment): running mean of squared gradients

        If exp_avg contains misalignment components, Adam will push the weights
        back toward misalignment even after we project the weights. By projecting
        exp_avg too, Adam forgets that forbidden direction in its momentum state.

        For exp_avg (momentum): we project it the same way as weights, removing
        the component along the misalignment direction. This is exact.

        For exp_avg_sq (variance): this stores element-wise squared gradient averages.
        Projecting it with the same linear operator is only a heuristic, so we
        keep that behavior optional.
        """
        if optimizer is None:
            return

        # Find the optimizer state for this parameter
        state = None
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                if p is param and p in optimizer.state:
                    state = optimizer.state[p]
                    break
            if state is not None:
                break

        if state is None:
            return

        with torch.no_grad():
            # Project first moment (momentum) — same linear projection as weights
            if 'exp_avg' in state and state['exp_avg'].shape == param.shape:
                state['exp_avg'].data = self._project_single_tensor(
                    state['exp_avg'].data, directions, cached_basis
                )

            # Optional heuristic: project second moment with the same operator.
            if (
                self.project_second_moment and
                'exp_avg_sq' in state and
                state['exp_avg_sq'].shape == param.shape
            ):
                state['exp_avg_sq'].data = self._project_single_tensor(
                    state['exp_avg_sq'].data, directions, cached_basis
                )

    def _apply_projection(self, model: torch.nn.Module, optimizer=None):
        """Apply projection to all mapped LoRA parameters and optionally optimizer states."""
        total_projection_norm = 0.0
        num_projected = 0
        debug_rows = []

        for layer_name, param in self._layer_to_param.items():
            directions = self.misalignment_subspace.get_directions_for_layer(layer_name)
            if not directions:
                continue

            cached = self._cached_bases.get(layer_name)

            with torch.no_grad():
                original_weight = param.data.clone()

                # Use batch projection for multiple directions (avoids Python loop)
                projected_weight = batch_project_out_subspace(
                    param.data, directions, _cached_basis=cached
                )

                # Apply projection strength
                if self.projection_strength < 1.0:
                    param.data = (
                        self.projection_strength * projected_weight +
                        (1.0 - self.projection_strength) * original_weight
                    )
                else:
                    param.data = projected_weight

                # Track statistics
                diff = original_weight - param.data
                total_projection_norm += torch.norm(diff).item()
                num_projected += 1

                if self.debug and self._step_count % self.debug_every_n_steps == 0:
                    weight_norm_before, blocked_before = _subspace_component_norm(
                        original_weight, directions, cached
                    )
                    weight_norm_after, blocked_after = _subspace_component_norm(
                        param.data, directions, cached
                    )
                    grad_stats = self._debug_grad_stats.get(layer_name, {})
                    debug_rows.append(
                        {
                            "layer_name": layer_name,
                            "grad_norm": grad_stats.get("grad_norm", 0.0),
                            "grad_blocked_norm": grad_stats.get("grad_blocked_norm", 0.0),
                            "weight_norm_before": weight_norm_before,
                            "blocked_before": blocked_before,
                            "weight_norm_after": weight_norm_after,
                            "blocked_after": blocked_after,
                            "removed_norm": torch.norm(diff).item(),
                        }
                    )

            # Project optimizer state too — this kills the push-pull
            if self.project_optimizer_states and optimizer is not None:
                self._project_optimizer_state_for_param(
                    param, directions, cached, optimizer
                )

        if self.verbose and num_projected > 0:
            avg_norm = total_projection_norm / num_projected
            print(f"[LoRAProjection] Step {self._step_count}: "
                  f"Projected {num_projected} layers, avg norm change: {avg_norm:.6f}"
                  f"{' (+ optimizer momentum' if self.project_optimizer_states else ''}"
                  f"{', second moment heuristic' if self.project_optimizer_states and self.project_second_moment else ''}"
                  f"{')' if self.project_optimizer_states else ''}")
        if debug_rows:
            self._print_debug_rows(debug_rows)

    def _print_debug_rows(self, rows: List[Dict[str, float]]):
        rows = sorted(rows, key=lambda row: row["blocked_before"], reverse=True)
        print(f"[LoRAProjection][Debug] Step {self._step_count}: showing top {min(len(rows), self.debug_max_layers)} layers")
        for row in rows[:self.debug_max_layers]:
            print(
                f"[LoRAProjection][Debug] {row['layer_name']}: "
                f"grad blocked {row['grad_blocked_norm']:.6f}/{row['grad_norm']:.6f} "
                f"({_format_ratio(row['grad_blocked_norm'], row['grad_norm'])}), "
                f"weight blocked {row['blocked_before']:.6f}/{row['weight_norm_before']:.6f} -> "
                f"{row['blocked_after']:.6f}/{row['weight_norm_after']:.6f} "
                f"(removed {row['removed_norm']:.6f})"
            )

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs
    ):
        """Capture gradient overlap with blocked directions before the optimizer step."""
        if model is None:
            return

        if not self._initialized:
            self._initialize_layer_mapping(model)

        upcoming_step = self._step_count + 1
        if upcoming_step % self.apply_every_n_steps != 0:
            return

        if self.debug and upcoming_step % self.debug_every_n_steps == 0:
            self._debug_grad_stats = {}
            for layer_name, param in self._layer_to_param.items():
                if param.grad is None:
                    continue
                directions = self.misalignment_subspace.get_directions_for_layer(layer_name)
                if not directions:
                    continue
                cached = self._cached_bases.get(layer_name)
                grad_norm, grad_blocked_norm = _subspace_component_norm(
                    param.grad, directions, cached
                )
                self._debug_grad_stats[layer_name] = {
                    "grad_norm": grad_norm,
                    "grad_blocked_norm": grad_blocked_norm,
                }

        self._project_gradients()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        optimizer=None,
        **kwargs
    ):
        """Called after each optimizer step to apply projection."""
        self._step_count += 1

        if model is None:
            return

        if not self._initialized:
            self._initialize_layer_mapping(model)

        if self._step_count % self.apply_every_n_steps != 0:
            return

        self._apply_projection(model, optimizer=optimizer)


# =============================================================================
# Shared Basis Projection (for RFA-style shared basis directions)
# =============================================================================


class ModuleSharedBasisProjectionCallback(TrainerCallback):
    """
    Project LoRA B weights against a module-wise shared basis.

    The basis is shared across all layers for a given module family
    (for example, all `self_attn.q_proj` B matrices share one basis).
    """

    def __init__(
        self,
        basis_by_module: Dict[str, torch.Tensor],
        projection_strength: float = 1.0,
        apply_every_n_steps: int = 1,
        verbose: bool = False,
        project_gradients: bool = False,
        gradient_projection_strength: float = 1.0,
        debug: bool = False,
        debug_every_n_steps: int = 1,
        debug_max_layers: int = 8,
    ):
        self.basis_by_module = basis_by_module
        self.projection_strength = projection_strength
        self.apply_every_n_steps = apply_every_n_steps
        self.verbose = verbose
        self.project_gradients = project_gradients
        self.gradient_projection_strength = gradient_projection_strength
        self.debug = debug
        self.debug_every_n_steps = debug_every_n_steps
        self.debug_max_layers = debug_max_layers
        self._step_count = 0
        self._initialized = False
        self._param_mapping: Dict[str, Tuple[torch.nn.Parameter, str]] = {}
        self._debug_grad_stats: Dict[str, Dict[str, float]] = {}

    def _get_short_name(self, full_name: str) -> Optional[str]:
        for module_name in self.basis_by_module:
            if module_name in full_name:
                return module_name
        return None

    def _initialize_mappings(self, model: torch.nn.Module):
        self._param_mapping = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_B' not in name or not name.endswith('.weight'):
                continue

            module_name = self._get_short_name(name)
            if module_name is None:
                continue
            self._param_mapping[name] = (param, module_name)
            if self.verbose:
                print(f"[SharedBasisProjection] Mapped {name} -> module basis[{module_name}]")

        if len(self._param_mapping) > 0:
            sample_param = next(iter(self._param_mapping.values()))[0]
            orthonormalized = {}
            for key, basis in self.basis_by_module.items():
                basis = basis.to(device=sample_param.device, dtype=sample_param.dtype)
                rows = orthonormalize_directions([row for row in basis])
                if len(rows) > 0:
                    orthonormalized[key] = torch.stack(rows, dim=0)
                else:
                    orthonormalized[key] = torch.zeros(
                        (1, basis.shape[-1]),
                        device=sample_param.device,
                        dtype=sample_param.dtype,
                    )
            self.basis_by_module = orthonormalized

        self._initialized = True
        print(f"[SharedBasisProjection] Initialized with {len(self._param_mapping)} parameters mapped")

    def _project_with_basis(self, weight: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        if basis.ndim == 1:
            basis = basis.unsqueeze(0)

        if basis.shape[1] == weight.shape[0]:
            coeffs = basis @ weight
            projection = basis.t() @ coeffs
            return weight - projection

        if basis.shape[1] == weight.numel():
            flat = weight.view(-1).float()
            basis_f = basis.float()
            coeffs = basis_f @ flat
            projection = basis_f.t() @ coeffs
            return (flat - projection).view_as(weight).to(weight.dtype)

        return weight

    def _basis_component_norm_for_weight(self, weight: torch.Tensor, basis: torch.Tensor) -> Tuple[float, float]:
        if basis.ndim == 1:
            basis = basis.unsqueeze(0)
        if basis.shape[1] == weight.shape[0]:
            return _basis_component_norm(weight, basis)
        if basis.shape[1] == weight.numel():
            return _subspace_component_norm(
                weight, [MisalignmentDirection(row.view(-1), "tmp") for row in basis]
            )
        total_norm = torch.norm(weight.detach().float()).item()
        return total_norm, 0.0

    def _project_gradients(self):
        if not self.project_gradients or self.gradient_projection_strength <= 0.0:
            return

        total_projection_norm = 0.0
        num_projected = 0

        for name, (param, module_name) in self._param_mapping.items():
            if param.grad is None:
                continue

            basis = self.basis_by_module.get(module_name)
            if basis is None or basis.numel() == 0:
                continue

            with torch.no_grad():
                original_grad = param.grad.data.clone()
                projected_grad = self._project_with_basis(param.grad.data, basis)
                if self.gradient_projection_strength < 1.0:
                    param.grad.data = (
                        self.gradient_projection_strength * projected_grad +
                        (1.0 - self.gradient_projection_strength) * original_grad
                    )
                else:
                    param.grad.data = projected_grad

                total_projection_norm += torch.norm(original_grad - param.grad.data).item()
                num_projected += 1

        if self.verbose and num_projected > 0:
            avg_norm = total_projection_norm / num_projected
            print(f"[SharedBasisProjection] Step {self._step_count + 1}: "
                  f"Projected gradients for {num_projected} params, avg norm change: {avg_norm:.6f}")

    def _apply_projection(self, model: torch.nn.Module):
        total_projection_norm = 0.0
        num_projected = 0
        debug_rows = []

        for name, (param, module_name) in self._param_mapping.items():
            basis = self.basis_by_module.get(module_name)
            if basis is None or basis.numel() == 0:
                continue

            with torch.no_grad():
                original_weight = param.data.clone()
                projected_weight = self._project_with_basis(param.data, basis)
                if self.projection_strength < 1.0:
                    param.data = (
                        self.projection_strength * projected_weight +
                        (1.0 - self.projection_strength) * original_weight
                    )
                else:
                    param.data = projected_weight

                diff = original_weight - param.data
                total_projection_norm += torch.norm(diff).item()
                num_projected += 1

                if self.debug and self._step_count % self.debug_every_n_steps == 0:
                    weight_norm_before, blocked_before = self._basis_component_norm_for_weight(original_weight, basis)
                    weight_norm_after, blocked_after = self._basis_component_norm_for_weight(param.data, basis)
                    grad_stats = self._debug_grad_stats.get(name, {})
                    debug_rows.append(
                        {
                            "layer_name": name,
                            "grad_norm": grad_stats.get("grad_norm", 0.0),
                            "grad_blocked_norm": grad_stats.get("grad_blocked_norm", 0.0),
                            "weight_norm_before": weight_norm_before,
                            "blocked_before": blocked_before,
                            "weight_norm_after": weight_norm_after,
                            "blocked_after": blocked_after,
                            "removed_norm": torch.norm(diff).item(),
                        }
                    )

        if self.verbose and num_projected > 0:
            avg_norm = total_projection_norm / num_projected
            print(f"[SharedBasisProjection] Step {self._step_count}: Projected {num_projected} params, avg norm change: {avg_norm:.6f}")
        if debug_rows:
            rows = sorted(debug_rows, key=lambda row: row["blocked_before"], reverse=True)
            print(f"[SharedBasisProjection][Debug] Step {self._step_count}: showing top {min(len(rows), self.debug_max_layers)} params")
            for row in rows[:self.debug_max_layers]:
                print(
                    f"[SharedBasisProjection][Debug] {row['layer_name']}: "
                    f"grad blocked {row['grad_blocked_norm']:.6f}/{row['grad_norm']:.6f} "
                    f"({_format_ratio(row['grad_blocked_norm'], row['grad_norm'])}), "
                    f"weight blocked {row['blocked_before']:.6f}/{row['weight_norm_before']:.6f} -> "
                    f"{row['blocked_after']:.6f}/{row['weight_norm_after']:.6f} "
                    f"(removed {row['removed_norm']:.6f})"
                )

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs
    ):
        if model is None:
            return
        if not self._initialized:
            self._initialize_mappings(model)

        upcoming_step = self._step_count + 1
        if upcoming_step % self.apply_every_n_steps != 0:
            return

        if self.debug and upcoming_step % self.debug_every_n_steps == 0:
            self._debug_grad_stats = {}
            for name, (param, module_name) in self._param_mapping.items():
                if param.grad is None:
                    continue
                basis = self.basis_by_module.get(module_name)
                if basis is None or basis.numel() == 0:
                    continue
                grad_norm, grad_blocked_norm = self._basis_component_norm_for_weight(param.grad, basis)
                self._debug_grad_stats[name] = {
                    "grad_norm": grad_norm,
                    "grad_blocked_norm": grad_blocked_norm,
                }

        self._project_gradients()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs
    ):
        self._step_count += 1
        if model is None:
            return
        if not self._initialized:
            self._initialize_mappings(model)
        if self._step_count % self.apply_every_n_steps != 0:
            return
        self._apply_projection(model)

def load_shared_basis_from_file(
    filepath: Union[str, Path],
    device: torch.device = None
) -> Dict:
    """
    Load shared basis directions from a file generated by generate_shared_basis.py.

    Expected file format:
    {
        'hidden_size': int,
        'rank': int,
        'n_modules': int,
        'module_names': list of module names,
        'layer_indices': list of layer indices,
        'basis_directions': tensor of shape [num_layers, rank, hidden_size]
    }

    Returns:
        Dict with the loaded data
    """
    if device is None:
        device = torch.device("cpu")

    data = torch.load(filepath, map_location=device, weights_only=True)

    # Validate format
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")

    if 'basis_directions' not in data:
        raise ValueError("File missing 'basis_directions' key. Is this a shared basis file?")

    return data


class SharedBasisProjectionCallback(TrainerCallback):
    """
    HuggingFace Trainer callback that projects out shared basis directions
    from LoRA adapter weights after each optimizer step.

    This is designed for use with shared basis directions computed across
    multiple misaligned LoRA models. The shared basis represents directions
    in activation space that are associated with misalignment.

    For LoRA training, we project these directions from the lora_B weights,
    which have shape (out_features, rank). For most modules, out_features
    equals hidden_size, so we can project the shared basis directions directly.
    """

    def __init__(
        self,
        basis_directions: torch.Tensor,
        layer_indices: List[int],
        module_names: List[str],
        projection_strength: float = 1.0,
        apply_every_n_steps: int = 1,
        verbose: bool = False,
        project_gradients: bool = False,
        gradient_projection_strength: float = 1.0,
        debug: bool = False,
        debug_every_n_steps: int = 1,
        debug_max_layers: int = 8,
    ):
        """
        Initialize the shared basis projection callback.

        Args:
            basis_directions: Tensor of shape [num_layers, rank, hidden_size]
            layer_indices: List of model layer indices that correspond to basis_directions
            module_names: List of module name patterns (e.g., ['self_attn.q_proj', ...])
            projection_strength: How much of the projection to apply (0.0 to 1.0)
            apply_every_n_steps: Apply projection every N optimizer steps
            verbose: If True, print projection statistics
            project_gradients: If True, project gradients before the optimizer step.
            gradient_projection_strength: How much of the gradient projection to apply.
            debug: If True, print per-layer gradient and weight projection diagnostics.
            debug_every_n_steps: Print diagnostics every N optimizer steps.
            debug_max_layers: Maximum number of parameters to print per diagnostic step.
        """
        self.basis_directions = basis_directions  # [num_layers, rank, hidden_size]
        self.layer_indices = layer_indices
        self.module_names = module_names
        self.projection_strength = projection_strength
        self.apply_every_n_steps = apply_every_n_steps
        self.verbose = verbose
        self.project_gradients = project_gradients
        self.gradient_projection_strength = gradient_projection_strength
        self.debug = debug
        self.debug_every_n_steps = debug_every_n_steps
        self.debug_max_layers = debug_max_layers
        self._step_count = 0
        self._initialized = False
        self._orthonormal_basis_directions = basis_directions.clone()
        # Map: param_name -> (param, basis_idx)
        self._param_mapping: Dict[str, tuple] = {}
        self._debug_grad_stats: Dict[str, Dict[str, float]] = {}

    def _get_short_name(self, full_name: str) -> Optional[str]:
        """Extract the module name from a full parameter name."""
        for module_name in self.module_names:
            if module_name in full_name:
                return module_name
        return None

    def _get_layer_idx(self, full_name: str) -> Optional[int]:
        """Extract the layer index from a full parameter name."""
        import re
        match = re.search(r'layers\.(\d+)\.', full_name)
        if match:
            return int(match.group(1))
        return None

    def _initialize_mappings(self, model: torch.nn.Module):
        """Build mapping from parameters to their corresponding basis directions."""
        self._param_mapping = {}

        # Build layer_idx -> basis_idx mapping
        layer_to_basis = {layer_idx: i for i, layer_idx in enumerate(self.layer_indices)}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Only process lora_B weights (they have hidden_size as first dimension)
            if 'lora_B' not in name or not name.endswith('.weight'):
                continue

            layer_idx = self._get_layer_idx(name)
            module_name = self._get_short_name(name)

            if layer_idx is None or module_name is None:
                continue

            # Check if this layer has basis directions
            if layer_idx not in layer_to_basis:
                continue

            basis_idx = layer_to_basis[layer_idx]

            # Store mapping
            self._param_mapping[name] = (param, basis_idx)
            if self.verbose:
                print(f"[SharedBasisProjection] Mapped {name} -> basis[{basis_idx}]")

        # Move basis directions to correct device/dtype
        if len(self._param_mapping) > 0:
            sample_param = next(iter(self._param_mapping.values()))[0]
            self.basis_directions = self.basis_directions.to(
                device=sample_param.device,
                dtype=sample_param.dtype
            )
            orthonormal_basis = []
            for basis in self.basis_directions:
                ortho_rows = orthonormalize_directions([row for row in basis])
                if len(ortho_rows) == 0:
                    ortho_rows = [torch.zeros_like(basis[0])]
                orthonormal_basis.append(torch.stack(ortho_rows, dim=0))
            self._orthonormal_basis_directions = orthonormal_basis

        self._initialized = True
        print(f"[SharedBasisProjection] Initialized with {len(self._param_mapping)} parameters mapped")

    def _project_out_basis(self, weight: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """
        Project out basis directions from a weight tensor using orthonormal projection.

        For lora_B with shape (out_features, lora_rank), we treat each column as a
        vector in out_features space and project out the basis subspace:

            W_proj = W - Bᵀ(BW)

        where B is (n_basis, hidden_size) with orthonormal rows.

        Args:
            weight: lora_B weight tensor of shape (out_features, lora_rank)
            basis: Basis directions of shape (n_basis, hidden_size) where hidden_size == out_features

        Returns:
            Projected weight tensor
        """
        out_features, lora_rank = weight.shape
        n_basis, hidden_size = basis.shape

        # Verify dimensions match
        if out_features != hidden_size:
            return weight

        # Orthonormal projection: W - Bᵀ(BW)
        coeffs = basis @ weight  # (n_basis, lora_rank)
        projection = basis.t() @ coeffs  # (hidden_size, lora_rank)

        return weight - projection

    def _project_gradients(self):
        if not self.project_gradients or self.gradient_projection_strength <= 0.0:
            return

        total_projection_norm = 0.0
        num_projected = 0

        for _, (param, basis_idx) in self._param_mapping.items():
            if param.grad is None:
                continue

            basis = self._orthonormal_basis_directions[basis_idx]
            with torch.no_grad():
                original_grad = param.grad.data.clone()
                projected_grad = self._project_out_basis(param.grad.data, basis)

                if self.gradient_projection_strength < 1.0:
                    param.grad.data = (
                        self.gradient_projection_strength * projected_grad +
                        (1.0 - self.gradient_projection_strength) * original_grad
                    )
                else:
                    param.grad.data = projected_grad

                total_projection_norm += torch.norm(original_grad - param.grad.data).item()
                num_projected += 1

        if self.verbose and num_projected > 0:
            avg_norm = total_projection_norm / num_projected
            print(f"[SharedBasisProjection] Step {self._step_count + 1}: "
                  f"Projected gradients for {num_projected} params, avg norm change: {avg_norm:.6f}")

    def _apply_projection(self, model: torch.nn.Module):
        """Apply projection to all mapped parameters."""
        total_projection_norm = 0.0
        num_projected = 0
        debug_rows = []

        for name, (param, basis_idx) in self._param_mapping.items():
            basis = self._orthonormal_basis_directions[basis_idx]  # [n_basis, hidden_size]

            with torch.no_grad():
                original_weight = param.data.clone()
                projected_weight = self._project_out_basis(param.data, basis)

                # Apply projection strength
                if self.projection_strength < 1.0:
                    param.data = (
                        self.projection_strength * projected_weight +
                        (1.0 - self.projection_strength) * original_weight
                    )
                else:
                    param.data = projected_weight

                # Track statistics
                diff = original_weight - param.data
                total_projection_norm += torch.norm(diff).item()
                num_projected += 1

                if self.debug and self._step_count % self.debug_every_n_steps == 0:
                    weight_norm_before, blocked_before = _basis_component_norm(original_weight, basis)
                    weight_norm_after, blocked_after = _basis_component_norm(param.data, basis)
                    grad_stats = self._debug_grad_stats.get(name, {})
                    debug_rows.append(
                        {
                            "layer_name": name,
                            "grad_norm": grad_stats.get("grad_norm", 0.0),
                            "grad_blocked_norm": grad_stats.get("grad_blocked_norm", 0.0),
                            "weight_norm_before": weight_norm_before,
                            "blocked_before": blocked_before,
                            "weight_norm_after": weight_norm_after,
                            "blocked_after": blocked_after,
                            "removed_norm": torch.norm(diff).item(),
                        }
                    )

        if self.verbose and num_projected > 0:
            avg_norm = total_projection_norm / num_projected
            print(f"[SharedBasisProjection] Step {self._step_count}: "
                  f"Projected {num_projected} params, avg norm change: {avg_norm:.6f}")
        if debug_rows:
            self._print_debug_rows(debug_rows)

    def _print_debug_rows(self, rows: List[Dict[str, float]]):
        rows = sorted(rows, key=lambda row: row["blocked_before"], reverse=True)
        print(f"[SharedBasisProjection][Debug] Step {self._step_count}: showing top {min(len(rows), self.debug_max_layers)} params")
        for row in rows[:self.debug_max_layers]:
            print(
                f"[SharedBasisProjection][Debug] {row['layer_name']}: "
                f"grad blocked {row['grad_blocked_norm']:.6f}/{row['grad_norm']:.6f} "
                f"({_format_ratio(row['grad_blocked_norm'], row['grad_norm'])}), "
                f"weight blocked {row['blocked_before']:.6f}/{row['weight_norm_before']:.6f} -> "
                f"{row['blocked_after']:.6f}/{row['weight_norm_after']:.6f} "
                f"(removed {row['removed_norm']:.6f})"
            )

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs
    ):
        """Capture gradient overlap with the shared basis before the optimizer step."""
        if model is None:
            return

        if not self._initialized:
            self._initialize_mappings(model)

        upcoming_step = self._step_count + 1
        if upcoming_step % self.apply_every_n_steps != 0:
            return

        if self.debug and upcoming_step % self.debug_every_n_steps == 0:
            self._debug_grad_stats = {}
            for name, (param, basis_idx) in self._param_mapping.items():
                if param.grad is None:
                    continue
                basis = self._orthonormal_basis_directions[basis_idx]
                grad_norm, grad_blocked_norm = _basis_component_norm(param.grad, basis)
                self._debug_grad_stats[name] = {
                    "grad_norm": grad_norm,
                    "grad_blocked_norm": grad_blocked_norm,
                }

        self._project_gradients()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs
    ):
        """Called after each optimizer step to apply projection."""
        self._step_count += 1

        if model is None:
            return

        if not self._initialized:
            self._initialize_mappings(model)

        if self._step_count % self.apply_every_n_steps != 0:
            return

        self._apply_projection(model)


def create_shared_basis_projection_callback(
    filepath: Union[str, Path],
    projection_strength: float = 1.0,
    apply_every_n_steps: int = 1,
    verbose: bool = False,
    project_gradients: bool = False,
    gradient_projection_strength: float = 1.0,
    debug: bool = False,
    debug_every_n_steps: int = 1,
    debug_max_layers: int = 8,
) -> SharedBasisProjectionCallback:
    """
    Create a SharedBasisProjectionCallback from a shared basis file.

    Args:
        filepath: Path to the shared basis .pt file
        projection_strength: How much projection to apply (0-1)
        apply_every_n_steps: Apply every N steps
        verbose: Print debug info
        project_gradients: Whether to project gradients before the optimizer step
        gradient_projection_strength: How much gradient projection to apply
        debug: Print per-layer gradient and weight diagnostics
        debug_every_n_steps: Print diagnostics every N steps
        debug_max_layers: Maximum number of params to print per diagnostic step

    Returns:
        Configured SharedBasisProjectionCallback
    """
    data = load_shared_basis_from_file(filepath)

    # Extract components
    basis_directions = data['basis_directions']  # [num_layers, rank, hidden_size]
    layer_indices = data.get('layer_indices', list(range(basis_directions.shape[0])))
    module_names = data.get('module_names', [
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
    ])

    print(f"[SharedBasisProjection] Loaded basis with shape {basis_directions.shape}")
    print(f"[SharedBasisProjection] Layer indices: {layer_indices}")
    print(f"[SharedBasisProjection] Modules: {module_names}")

    return SharedBasisProjectionCallback(
        basis_directions=basis_directions,
        layer_indices=layer_indices,
        module_names=module_names,
        projection_strength=projection_strength,
        apply_every_n_steps=apply_every_n_steps,
        verbose=verbose,
        project_gradients=project_gradients,
        gradient_projection_strength=gradient_projection_strength,
        debug=debug,
        debug_every_n_steps=debug_every_n_steps,
        debug_max_layers=debug_max_layers,
    )
