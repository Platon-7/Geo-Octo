from pathlib import Path

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec

REPO_ROOT = Path(__file__).resolve().parents[3]


def get_config(config_string="full,multimodal"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    UNIFIED_STATS_PATH = str(
        REPO_ROOT
        / "libero_datasets"
        / "unified_stats"
        / "unified_dataset_statistics_libero_object_no_vggt.json"
    )

    FINETUNING_KWARGS = {
        "name": "libero_object_no_noops",
        "data_dir": str(REPO_ROOT / "libero_datasets"),
        "dataset_statistics": UNIFIED_STATS_PATH,
        "image_obs_keys": {"primary": "image_primary"},
        "proprio_obs_key": "proprio",
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        "action_normalization_mask": [True, True, True, True, True, True, False],
        "standardize_fn": ModuleSpec.create(
            "octo.data.utils.data_utils:standardize_libero_vggt"
        ),
        "num_parallel_reads": 8,
        "num_parallel_calls": 16,
    }

    # PointMap finetuning + LoRA: freeze transformer backbone; unfreeze heads, projections, norm adapters
    frozen_keys = (
        # Tokenizers and projections/norm adapters
        "octo_transformer.observation_tokenizers_*",
        "octo_transformer.task_tokenizers_*",
        "octo_transformer.task_*",
        # NOTE: Projections and norm adapters are intentionally NOT frozen
        # Transformer backbone fully frozen (we'll override LoRA params as trainable)
        "octo_transformer.BlockTransformer_*",
        # Positional embeddings
        "octo_transformer.*_pos_embedding",
        # Heads are intentionally NOT frozen
    )

    max_steps = FieldReference(50000)
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
        save_interval=5000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_finetune_pointmap", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
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
            # Ensure LoRA/pointmap stay trainable despite broad freezes (harmless if already unfrozen)
            trainable_overrides=(
                # Readout bottleneck fusion + gates for pointmap injection
                "octo_transformer.readout_*_bottleneck_*",
                "octo_transformer.readout_*_pointmap_gate",
                # Pointmap encoder
                "octo_transformer.pointmap_encoder*",
                # LoRA matrices inside transformer attention/MLP
                "octo_transformer.BlockTransformer_0.Transformer_0.encoderblock_*.*.lora_*.*",
            ),
            grad_accumulation_steps=4,
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=0,
            trajs_for_viz=0,
            samples_per_state=0,
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
            "primary": (256, 256),
            "wrist": (128, 128),
        },
        image_augment_kwargs=dict(
            primary=workspace_augment_kwargs,
        ),
    )

    config["frame_transform_threads"] = 24
    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    config["config_delete_keys"] = {"model": {"observation_tokenizers": {"wrist": True}}}

    # Add pointmap wiring only (the encoder spec is created in the finetuning script to match token_embedding_size)
    config["update_config"] = {
        "model": {
            "pointmap_input_key": "pointmap",
            # Enable LoRA in transformer blocks with sensible defaults
            "transformer_kwargs": {
                "use_lora_attention": True,
                "use_lora_mlp": True,
                "lora_r": 8,
                "lora_alpha": 16.0,
                "lora_dropout": 0.0,
                "lora_attn_q": True,
                "lora_attn_k": False,
                "lora_attn_v": True,
                "lora_attn_out": False,
            },
        }
    }

    return ConfigDict(config)