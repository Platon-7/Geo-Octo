# scripts/configs/config_offline.py (Final version)
from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
from octo.utils.spec import ModuleSpec
import numpy as np

def get_config(config_string="full,multimodal"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Use ALL your VGGT datasets for training
    DATASET_KWARGS_LIST = [
        {
            "name": "libero_object_vggt",
            "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2",
            "standardize_fn": ModuleSpec.create(
                "octo.data.utils.data_utils:standardize_libero_vggt"
            ),
            "image_obs_keys": {"primary": "image_primary"},
            "proprio_obs_key": "proprio",
            "language_key": "language_instruction",
            "action_proprio_normalization_type": "normal",
            "filter_functions": [],
        },
        {
            "name": "libero_spatial_vggt",
            "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2",
            "standardize_fn": ModuleSpec.create(
                "octo.data.utils.data_utils:standardize_libero_vggt"
            ),
            "image_obs_keys": {"primary": "image_primary"},
            "proprio_obs_key": "proprio",
            "language_key": "language_instruction",
            "action_proprio_normalization_type": "normal",
            "filter_functions": [],
        },
        {
            "name": "libero_goal_vggt",  # Added the missing dataset
            "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2",
            "standardize_fn": ModuleSpec.create(
                "octo.data.utils.data_utils:standardize_libero_vggt"
            ),
            "image_obs_keys": {"primary": "image_primary"},
            "proprio_obs_key": "proprio",
            "language_key": "language_instruction",
            "action_proprio_normalization_type": "normal",
            "filter_functions": [],
        },
        {
            "name": "liber_o10_vggt",
            "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2",
            "standardize_fn": ModuleSpec.create(
                "octo.data.utils.data_utils:standardize_libero_vggt"
            ),
            "image_obs_keys": {"primary": "image_primary"},
            "proprio_obs_key": "proprio",
            "language_key": "language_instruction",
            "action_proprio_normalization_type": "normal",
            "filter_functions": [],
        },
    ]

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    else: # head_mlp_only
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=1)  # Reduce window size to save memory

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=1,  # Reduce from 2 to 1 to save memory
        shuffle_buffer_size=5,  # Reduce shuffle buffer to save memory
        num_steps=50000,
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_vggt_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs_list=DATASET_KWARGS_LIST,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=1e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=4,  # Accumulate gradients over 4 steps to simulate batch_size=4
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=100,  # Reduce validation buffer
            num_val_batches=4,  # Reduce validation batches
        ),
        viz_kwargs=dict(
            eval_batch_size=16,  # Reduce eval batch size significantly
            trajs_for_metrics=20,  # Reduce trajectories for metrics
            trajs_for_viz=2,  # Reduce visualization trajectories
            samples_per_state=2,  # Reduce samples per state
        ),
    )

    if task == "multimodal":
        task_augment_strategy="delete_task_conditioning"
        task_augment_kwargs=dict(keep_image_prob=0.5)
    else:
        task_augment_strategy=None
        task_augment_kwargs={}

    config["traj_transform_kwargs"] = dict(
        window_size=window_size,
        action_horizon=window_size,
        task_augment_strategy=task_augment_strategy,
        task_augment_kwargs=task_augment_kwargs,
    )
    
    # Add frame transform kwargs to fix the resize_size warning
    config["frame_transform_kwargs"] = dict(
        resize_size={
            "primary": (256, 256),  # Match original Libero image size
        },
        image_augment_kwargs=dict(
            primary=dict(
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.1],
                random_contrast=[0.9, 1.1],
                random_saturation=[0.9, 1.1],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness", 
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            ),
        ),
    )
    
    config['update_config'] = {"model": {"observation_tokenizers": {"vggt_tokens": ModuleSpec.create('octo.model.components.tokenizers:VGGTTokenizer')}}}

    return ConfigDict(config)