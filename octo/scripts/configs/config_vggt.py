from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
from octo.model.components.tokenizers import VGGTTokenizer, ImageTokenizer, VisionMixer
from octo.utils.spec import ModuleSpec
from octo.model.components.vit_encoders import PatchEncoder


def get_config(config_string="full,multimodal"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    UNIFIED_STATS_PATH = "/home/pkarageorgis/geo_octo/libero_datasets/unified_stats/unified_dataset_statistics_libero_spatial_vggt_compressed_24_torch_ae.json"

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)

    FINETUNING_KWARGS = {
        "name": "libero_spatial_vggt_compressed_torch",
        "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_onnx_compressed_torch_ae",
        "dataset_statistics": UNIFIED_STATS_PATH,
        "image_obs_keys": {"primary": "image_primary"},
        "proprio_obs_key": "proprio",
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        # We want to avoid normalizing the gripper
        "action_normalization_mask": [True, True, True, True, True, True, False],
        # standardize_fn is dynamically loaded from a file
        "standardize_fn": ModuleSpec.create(
                "octo.data.utils.data_utils:standardize_libero_vggt"
        ),
        # If the default data loading speed is too slow, try these:
        # "num_parallel_reads": 8,  # for reading from disk / GCS
        # "num_parallel_calls": 16,  # for initial dataset construction
    }

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

    max_steps = FieldReference(200000)
    window_size = FieldReference(default=1)

    config = dict(
        resume_dir="",
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=64,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=1000,
        save_interval=1000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_finetune_vggt", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
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
            grad_accumulation_steps=4,  # if you are using grad accumulation, you need to adjust max_steps accordingly
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
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
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
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),  # wrist camera is at 128x128
        },
        image_augment_kwargs={},
    )
    
    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    
    
#     config['update_config'] = {
#     "model": {
#         "observation_tokenizers": {
#             "mixed_vision": ModuleSpec.create(
#                 "octo.model.components.tokenizers:VisionMixer",
#                 patch_tokenizer_spec={
#                     "module": "octo.model.components.tokenizers:ImageTokenizer",
#                     "kwargs": {
#                         "encoder": ModuleSpec.create(
#                             "octo.model.components.vit_encoders:PatchEncoder",
#                             patch_size=32,
#                             num_features=512,
#                         ),
#                         "obs_stack_keys": ("image_primary",),
#                     },
#                 },
#                 vggt_tokenizer_spec={
#                     "module": "octo.model.components.tokenizers:VGGTTokenizer",
#                 },
#             ),
#         },
#         "repeat_task_tokens": True,
#     }
# }
    
    
    config['config_delete_keys'] = {
        "model": {"observation_tokenizers": {"wrist": True}}
    }
    
    # config['config_delete_keys'] = {
    #     "model": {
    #         "observation_tokenizers": True  # remove the entire dict from the base config
    #     }
    # }
    

#     # VGGT ONLY VERSION
#     config['update_config'] = {
#     "model": {
#         "observation_tokenizers": {
#             # Keep the same key so downstream code keeps working
#             "mixed_vision": ModuleSpec.create(
#                 "octo.model.components.tokenizers:VGGTTokenizer"
#             ),
#         },
#         "repeat_task_tokens": False,
#     }
# }

    
    return ConfigDict(config)