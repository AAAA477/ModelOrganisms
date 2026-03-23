import os
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainingConfig(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    # Required model and data paths
    model: str = Field(..., description="Hugging Face model ID")
    training_file: str = Field(..., description="File ID of the training dataset")
    test_file: Optional[str] = Field(None, description="File ID of the test dataset")

    # Output model
    finetuned_model_id: str = Field('{org_id}/{model_name}-{job_id}', description="File ID of the finetuned model")

    # Model configuration
    max_seq_length: int = Field(2048, description="Maximum sequence length for training")
    load_in_4bit: bool = Field(False, description="Whether to load model in 4-bit quantization")

    # Training type configuration
    loss: Literal["dpo", "orpo", "sft"] = Field(..., description="Loss function / training type")

    # PEFT configuration
    is_peft: bool = Field(True, description="Whether to use PEFT for training")
    target_modules: Optional[List[str]] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target modules for LoRA"
    )
    layers_to_transform: Optional[List[int]] = Field(
        None,
        description="Layers to transform for LoRA. If None, all layers will be transformed."
    )
    lora_bias: Literal["all", "none"] = Field("none", description="Value for FastLanguageModel.get_peft_model(bias=?)")

    # LoRA specific arguments
    r: int = Field(16, description="LoRA attention dimension")
    lora_alpha: int = Field(16, description="LoRA alpha parameter")
    lora_dropout: float = Field(0.0, description="LoRA dropout rate")
    use_rslora: bool = Field(True, description="Whether to use RSLoRA")
    merge_before_push: bool = Field(True, description="Whether to merge model before pushing to Hub. Only merged models can be used as parent models for further finetunes. Only supported for bf16 models.")
    push_to_private: bool = Field(True, description="Whether to push to private Hub")
    push_only_adapters: bool = Field(False, description="Whether to push only the LoRA adapters to Hub instead of the full model")

    # Training hyperparameters
    epochs: int = Field(1, description="Number of training epochs")
    max_steps: Optional[int] = Field(None, description="Maximum number of training steps")
    per_device_train_batch_size: int = Field(2, description="Training batch size per device")
    gradient_accumulation_steps: int = Field(8, description="Number of gradient accumulation steps")
    warmup_steps: int = Field(5, description="Number of warmup steps")
    learning_rate: Union[float, str] = Field(1e-4, description="Learning rate or string expression")
    logging_steps: int = Field(1, description="Number of steps between logging")
    evaluation_steps: int = Field(50, description="Number of steps between evaluations on the test set")
    optim: str = Field("adamw_8bit", description="Optimizer to use for training")
    weight_decay: float = Field(0.01, description="Weight decay rate")
    lr_scheduler_type: str = Field("linear", description="Learning rate scheduler type")
    seed: int = Field(3407, description="Random seed for reproducibility")
    beta: float = Field(0.1, description="Beta parameter for DPO/ORPO training")
    save_steps: int = Field(5000, description="Save checkpoint every X steps")
    output_dir: str = Field("./tmp", description="Output directory for training checkpoints")
    train_on_responses_only: bool = Field(False, description="Whether to train on responses only")

    # Weight Projection (Misalignment Prevention) configuration
    enable_weight_projection: bool = Field(
        False,
        description="Enable weight projection to prevent emergent misalignment during training"
    )
    misalignment_direction_path: Optional[str] = Field(
        None,
        description="Path to .pt file containing misalignment direction tensors for weight projection"
    )
    misalignment_lora_repos: Optional[List[str]] = Field(
        None,
        description="List of HuggingFace LoRA repo IDs to compute misalignment directions from"
    )
    average_misalignment_directions: bool = Field(
        True,
        description="If True, average directions from multiple LoRA models into one direction per layer. "
                    "If False, keep all directions separately and project out each individually."
    )
    projection_strength: float = Field(
        1.0,
        description="Strength of weight projection (0.0 to 1.0). 1.0 = full projection"
    )
    projection_every_n_steps: int = Field(
        1,
        description="Apply weight projection every N optimizer steps"
    )
    projection_target_modules: Optional[List[str]] = Field(
        None,
        description="Target modules for weight projection. If None, applies to all registered layers"
    )
    projection_verbose: bool = Field(
        False,
        description="Print verbose statistics during weight projection"
    )
    project_optimizer_states: bool = Field(
        True,
        description="If True (default), also project out misalignment components from Adam's "
                    "first-moment momentum (exp_avg) after each weight projection step. "
                    "This prevents the optimizer from 'remembering' the misalignment direction and "
                    "pushing weights back toward it on subsequent steps. Only applies when using "
                    "weight projection (not gradient projection)."
    )
    project_second_moment: bool = Field(
        False,
        description="If True, also project Adam's second-moment estimate (exp_avg_sq). "
                    "This is a heuristic rather than an exact projection because squared-gradient "
                    "statistics do not live in the same linear space as weights."
    )
    projection_debug: bool = Field(
        False,
        description="If True, print debug statistics showing how much of each gradient and weight "
                    "lies in the blocked projection subspace."
    )
    projection_debug_every_n_steps: int = Field(
        1,
        description="When projection_debug is enabled, print debug stats every N optimizer steps."
    )
    projection_debug_max_layers: int = Field(
        8,
        description="Maximum number of layers to print per projection debug report."
    )

    @model_validator(mode="before")
    def validate_training_file_prefixes(cls, values):
        loss = values.get('loss', 'orpo')
        training_file = values.get('training_file')

        if os.path.exists(training_file):
            return values

        # if loss == 'sft' and not training_file.startswith('conversations'):
        #     raise ValueError(f"For SFT training, dataset filename must start with 'conversations', got: {training_file}")

        if loss in ['dpo', 'orpo'] and not training_file.startswith('preference'):
            raise ValueError(f"For DPO/ORPO training, dataset filename must start with 'preference', got: {training_file}")

        return values

    @field_validator("finetuned_model_id")
    def validate_finetuned_model_id(cls, v):
        # if v and model_exists(v):
        #     raise ValueError(f"Model {v} already exists")
        if len(v.split("/")) != 2:
            raise ValueError("Model ID must be in the format 'user/model'")
        org, model = v.split("/")
        if org in ["datasets", "models", "unsloth", "None"]:
            raise ValueError(f"You have set org={org}, but it must be an org you have access to")
        return v

    @field_validator("learning_rate", mode="before")
    def validate_learning_rate(cls, v):
        if isinstance(v, float) and v <= 0:
            raise ValueError("Learning rate must be positive")
        return v

    @field_validator("lora_dropout")
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        return v

    @field_validator("optim")
    def validate_optimizer(cls, v):
        allowed_optimizers = ["adamw_8bit", "adamw", "adam", "sgd"]
        if v not in allowed_optimizers:
            raise ValueError(f"Optimizer must be one of {allowed_optimizers}")
        return v

    @field_validator("lr_scheduler_type")
    def validate_scheduler(cls, v):
        allowed_schedulers = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        if v not in allowed_schedulers:
            raise ValueError(f"Scheduler must be one of {allowed_schedulers}")
        return v

    @field_validator("projection_strength")
    def validate_projection_strength(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Projection strength must be between 0.0 and 1.0")
        return v

    @field_validator("projection_every_n_steps")
    def validate_projection_every_n_steps(cls, v):
        if v < 1:
            raise ValueError("projection_every_n_steps must be at least 1")
        return v

    @field_validator("projection_debug_every_n_steps")
    def validate_projection_debug_every_n_steps(cls, v):
        if v < 1:
            raise ValueError("projection_debug_every_n_steps must be at least 1")
        return v

    @field_validator("projection_debug_max_layers")
    def validate_projection_debug_max_layers(cls, v):
        if v < 1:
            raise ValueError("projection_debug_max_layers must be at least 1")
        return v

    @model_validator(mode="after")
    def validate_weight_projection_config(self):
        """Validate weight projection configuration consistency."""
        if self.enable_weight_projection:
            if self.misalignment_direction_path is None and self.misalignment_lora_repos is None:
                raise ValueError(
                    "When enable_weight_projection is True, you must provide either "
                    "misalignment_direction_path or misalignment_lora_repos"
                )
            if self.misalignment_direction_path is not None and self.misalignment_lora_repos is not None:
                raise ValueError(
                    "Cannot specify both misalignment_direction_path and misalignment_lora_repos. "
                    "Choose one source for misalignment directions."
                )
        return self
