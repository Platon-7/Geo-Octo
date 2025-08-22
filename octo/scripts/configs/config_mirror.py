from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec


def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]
    UNIFIED_STATS_PATH = "/home/pkarageorgis/geo_octo/libero_datasets/unified_stats/unified_dataset_statistics_libero_spatial_no_vggt.json"
    
    

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)

    DATASET_KWARGS_LIST = [
        {
            "name": "libero_spatial_no_noops",
            "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_datasets",
            "dataset_statistics": UNIFIED_STATS_PATH,  # Use the unified file
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
        frozen_keys = ("octo_transformer.BlockTransformer_*",)
    else: # head_mlp_only
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )


    max_steps = FieldReference(150000)
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=128,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_finetune_mirror", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs_list=DATASET_KWARGS_LIST,
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
            grad_accumulation_steps=2,  # if you are using grad accumulation, you need to adjust max_steps accordingly
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

    config["traj_transform_kwargs"] = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
    )

    config["frame_transform_kwargs"] = dict(
    resize_size={
        "primary": (256, 256),
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
    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    return ConfigDict(config)