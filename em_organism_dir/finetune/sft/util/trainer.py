from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from em_organism_dir.finetune.sft.util.weight_projection import (
    WeightProjectionCallback,
    LoRAProjectionCallback,
    ModuleSharedBasisProjectionCallback,
    load_misalignment_directions_from_file,
    create_lora_subspace_from_lora_models,
    create_module_shared_basis_from_lora_models,
    create_shared_basis_projection_callback,
)


def create_weight_projection_callback(training_cfg):
    """
    Create a projection callback based on the training configuration.

    Callback selection depends on both the training mode and the direction source:
    - PEFT / LoRA training uses LoRA-space callbacks operating on lora_A/lora_B
    - full finetuning uses WeightProjectionCallback on full trainable weights
    - shared-basis files use SharedBasisProjectionCallback

    Args:
        training_cfg: TrainingConfig object with weight projection settings

    Returns:
        A configured projection callback if enabled, otherwise None
    """
    if not training_cfg.enable_weight_projection:
        return None

    is_peft = getattr(training_cfg, 'is_peft', True)
    common_debug_kwargs = dict(
        debug=training_cfg.projection_debug,
        debug_every_n_steps=training_cfg.projection_debug_every_n_steps,
        debug_max_layers=training_cfg.projection_debug_max_layers,
    )

    # Load misalignment directions from the specified source
    if training_cfg.misalignment_direction_path:
        print(f"[WeightProjection] Loading directions from: {training_cfg.misalignment_direction_path}")

        # Check if this is a shared basis file (has 'basis_directions' key)
        import torch
        data = torch.load(training_cfg.misalignment_direction_path, map_location='cpu', weights_only=True)

        if isinstance(data, dict) and 'basis_directions' in data:
            # Use SharedBasisProjectionCallback for shared basis files
            print("[WeightProjection] Detected shared basis format, using SharedBasisProjectionCallback")
            callback = create_shared_basis_projection_callback(
                training_cfg.misalignment_direction_path,
                projection_strength=training_cfg.projection_strength,
                apply_every_n_steps=training_cfg.projection_every_n_steps,
                verbose=training_cfg.projection_verbose,
                project_gradients=getattr(training_cfg, 'project_gradients', False),
                gradient_projection_strength=getattr(training_cfg, 'gradient_projection_strength', 1.0),
                **common_debug_kwargs,
            )
            num_layers = data['basis_directions'].shape[0]
            print(f"[SharedBasisProjection] Loaded directions for {num_layers} basis layers")
            print("[SharedBasisProjection] Projection callback enabled for training")
            return callback
        elif isinstance(data, dict) and all(hasattr(v, "shape") for v in data.values()):
            print("[WeightProjection] Detected module-wise shared basis format, using ModuleSharedBasisProjectionCallback")
            callback = ModuleSharedBasisProjectionCallback(
                basis_by_module=data,
                projection_strength=training_cfg.projection_strength,
                apply_every_n_steps=training_cfg.projection_every_n_steps,
                verbose=training_cfg.projection_verbose,
                project_gradients=getattr(training_cfg, 'project_gradients', False),
                gradient_projection_strength=getattr(training_cfg, 'gradient_projection_strength', 1.0),
                **common_debug_kwargs,
            )
            print(f"[SharedBasisProjection] Loaded module bases for {len(data)} module types")
            print("[SharedBasisProjection] Projection callback enabled for training")
            return callback
        else:
            # Use the appropriate callback for regular direction files
            subspace = load_misalignment_directions_from_file(
                training_cfg.misalignment_direction_path
            )
            layer_names = subspace.get_all_layer_names()
            looks_like_lora_space = all(
                (".lora_A" in layer_name) or (".lora_B" in layer_name)
                for layer_name in layer_names
            )

            if is_peft:
                if not looks_like_lora_space:
                    raise ValueError(
                        "LoRA training requires LoRA-space projection directions "
                        "(keys containing .lora_A or .lora_B) when loading from "
                        "misalignment_direction_path. Use a shared basis file, "
                        "provide LoRA-space directions, or set misalignment_lora_repos."
                    )

                project_opt_states = getattr(training_cfg, 'project_optimizer_states', True)
                project_gradients = getattr(training_cfg, 'project_gradients', False)
                gradient_projection_strength = getattr(training_cfg, 'gradient_projection_strength', 1.0)
                project_second_moment = getattr(training_cfg, 'project_second_moment', False)
                callback = LoRAProjectionCallback(
                    misalignment_subspace=subspace,
                    projection_strength=training_cfg.projection_strength,
                    apply_every_n_steps=training_cfg.projection_every_n_steps,
                    verbose=training_cfg.projection_verbose,
                    project_optimizer_states=project_opt_states,
                    project_gradients=project_gradients,
                    gradient_projection_strength=gradient_projection_strength,
                    project_second_moment=project_second_moment,
                    **common_debug_kwargs,
                )
            else:
                callback = WeightProjectionCallback(
                    misalignment_subspace=subspace,
                    projection_strength=training_cfg.projection_strength,
                    apply_every_n_steps=training_cfg.projection_every_n_steps,
                    target_modules=training_cfg.projection_target_modules,
                    verbose=training_cfg.projection_verbose,
                    **common_debug_kwargs,
                )
    elif training_cfg.misalignment_lora_repos:
        if not is_peft:
            raise ValueError(
                "misalignment_lora_repos builds a LoRA-space projection subspace and "
                "only makes sense when is_peft=True."
            )
        if getattr(training_cfg, 'use_shared_basis_from_lora_repos', False):
            print(f"[SharedBasisProjection] Computing module shared basis from {len(training_cfg.misalignment_lora_repos)} LoRA models")
            basis_by_module = create_module_shared_basis_from_lora_models(
                training_cfg.misalignment_lora_repos,
                target_modules=training_cfg.projection_target_modules,
                quiet=not training_cfg.projection_verbose,
                target_rank=training_cfg.shared_basis_rank,
            )
            callback = ModuleSharedBasisProjectionCallback(
                basis_by_module=basis_by_module,
                projection_strength=training_cfg.projection_strength,
                apply_every_n_steps=training_cfg.projection_every_n_steps,
                verbose=training_cfg.projection_verbose,
                project_gradients=getattr(training_cfg, 'project_gradients', False),
                gradient_projection_strength=getattr(training_cfg, 'gradient_projection_strength', 1.0),
                **common_debug_kwargs,
            )
            print(f"[SharedBasisProjection] Created module shared basis for {len(basis_by_module)} module types")
            return callback
        print(f"[LoRAProjection] Computing directions from {len(training_cfg.misalignment_lora_repos)} LoRA models")
        print(f"[LoRAProjection] Averaging directions: {training_cfg.average_misalignment_directions}")

        # Use the new LoRA-space projection
        subspace = create_lora_subspace_from_lora_models(
            training_cfg.misalignment_lora_repos,
            target_modules=training_cfg.projection_target_modules,
            quiet=not training_cfg.projection_verbose,
            average_directions=training_cfg.average_misalignment_directions
        )

        project_opt_states = getattr(training_cfg, 'project_optimizer_states', True)
        project_gradients = getattr(training_cfg, 'project_gradients', False)
        gradient_projection_strength = getattr(training_cfg, 'gradient_projection_strength', 1.0)
        project_second_moment = getattr(training_cfg, 'project_second_moment', False)
        print(f"[LoRAProjection] Using LoRA-space weight projection (operating on A and B matrices directly)")
        print(f"[LoRAProjection] Project optimizer states: {project_opt_states}")
        print(f"[LoRAProjection] Project gradients: {project_gradients}")
        if project_opt_states:
            print(f"[LoRAProjection] Project second moment (heuristic): {project_second_moment}")
        callback = LoRAProjectionCallback(
            misalignment_subspace=subspace,
            projection_strength=training_cfg.projection_strength,
            apply_every_n_steps=training_cfg.projection_every_n_steps,
            verbose=training_cfg.projection_verbose,
            project_optimizer_states=project_opt_states,
            project_gradients=project_gradients,
            gradient_projection_strength=gradient_projection_strength,
            project_second_moment=project_second_moment,
            **common_debug_kwargs,
        )
    else:
        raise ValueError(
            "Weight projection enabled but no direction source specified. "
            "Set either misalignment_direction_path or misalignment_lora_repos."
        )

    num_layers = len(subspace.get_all_layer_names())
    print(f"[LoRAProjection] Loaded directions for {num_layers} layers")

    return callback


def get_instruct_response_part(tokenizer):
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True, tokenize=False).replace(example_text, '')
    return instruction_part, response_part


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    # NOTE: maybe this is not needed but we should test it with train_on_responses_only: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            texts.append(
                tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                ) + tokenizer.eos_token
            )
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)

    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    import wandb
    wandb.init(
        project="clarifying-em",
        name=training_cfg.finetuned_model_id,  # Uses model ID as run name
        config=training_cfg
    )

    # Create callbacks list, including weight projection if enabled
    callbacks = []
    weight_projection_callback = create_weight_projection_callback(training_cfg)
    if weight_projection_callback is not None:
        callbacks.append(weight_projection_callback)
        print(f"[{weight_projection_callback.__class__.__name__}] Projection callback enabled for training")

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        #dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            warmup_steps=training_cfg.warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=training_cfg.optim,
            weight_decay=training_cfg.weight_decay,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            seed=training_cfg.seed,
            report_to=["wandb"],
            num_train_epochs=training_cfg.epochs,
            push_to_hub=True,
            hub_model_id=training_cfg.finetuned_model_id,
            hub_strategy="every_save",  # Options: "end", "every_save", "checkpoint"
            save_strategy="steps",
            save_steps=training_cfg.save_steps,
            output_dir=training_cfg.output_dir,
            eval_steps=training_cfg.evaluation_steps,
            do_eval=True,
            eval_strategy="steps",
            **kwargs,
        ),
        callbacks=callbacks,
        eval_dataset=test_dataset,
    )

    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer = tokenizer)
        trainer = train_on_responses_only(
            SFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)
    return trainer
