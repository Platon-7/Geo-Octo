from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec
from octo.model.components.tokenizers import VGGTTokenizer, VisionMixer, ImageTokenizer
from octo.model.components.vit_encoders import PatchEncoder


def get_config(config_string="full,multimodal"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Configuration for VGGT datasets
    VGGT_DATASET_KWARGS_LIST = [
        {
            "name": "libero_object_vggt",
            "data_dir": "/workspace/libero_vggt_datasets2",  # Update this to your actual path
            "image_obs_keys": {"primary": "image"},
            "proprio_obs_key": "proprio",
            "language_key": None,  # Set if you have language instructions
            "action_proprio_normalization_type": "normal",
            "action_normalization_mask": [True, True, True, True, True, True, False],
            "standardize_fn": None,  # No standardization needed since data is already processed
        },
        {
            "name": "libero_spatial_vggt", 
            "data_dir": "/workspace/libero_vggt_datasets2",
            "image_obs_keys": {"primary": "image"},
            "proprio_obs_key": "proprio",
            "language_key": None,
            "action_proprio_normalization_type": "normal",
            "action_normalization_mask": [True, True, True, True, True, True, False],
            "standardize_fn": None,
        },
        {
            "name": "libero_goal_vggt",
            "data_dir": "/workspace/libero_vggt_datasets2", 
            "image_obs_keys": {"primary": "image"},
            "proprio_obs_key": "proprio",
            "language_key": None,
            "action_proprio_normalization_type": "normal",
            "action_normalization_mask": [True, True, True, True, True, True, False],
            "standardize_fn": None,
        },
        {
            "name": "liber_o10_vggt",  # Note: this matches the typo in the original dataset
            "data_dir": "/workspace/libero_vggt_datasets2",
            "image_obs_keys": {"primary": "image"},
            "proprio_obs_key": "proprio", 
            "language_key": None,
            "action_proprio_normalization_type": "normal",
            "action_normalization_mask": [True, True, True, True, True, True, False],
            "standardize_fn": None,
        },
    ]

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=8,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_vggt_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs_list=VGGT_DATASET_KWARGS_LIST,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
    )
    
    workspace_augment_kwargs = dict(
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
    )
    
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (224, 224),  # Match VGGT input size
        },
        image_augment_kwargs=dict(
            primary=workspace_augment_kwargs,
        ),
    )
    
    config["frame_transform_threads"] = 16
    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs

    # Model configuration update for VGGT tokens
    config['update_config'] = {
        "model": {
            "observation_tokenizers": {
                "vggt_tokens": ModuleSpec.create(VGGTTokenizer),
                # Keep the primary image tokenizer for mixed vision
                "image_primary": ModuleSpec.create(
                    ImageTokenizer,
                    encoder=ModuleSpec.create(
                        PatchEncoder,
                        patch_size=16,
                        num_features=512
                    ),
                    obs_stack_keys=("image_primary",),
                ),
            }
        }
    }

    return ConfigDict(config)