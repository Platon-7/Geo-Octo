# scripts/configs/config_offline.py (Final version)
from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
from octo.utils.spec import ModuleSpec
import numpy as np

def get_config(config_string="full,multimodal"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    def standardize_libero_vggt(traj: dict) -> dict:
        traj['observation']['image_primary'] = traj['observation'].pop('image')
        traj['observation']['proprio'] = traj['observation'].pop('state')
        return traj

    DATASET_KWARGS = {
        # This can be any of your new datasets, e.g., "libero_10_no_noops_vggt"
        "name": "libero_10_no_noops_vggt",
        
        "data_dir": "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2", 

        "standardize_fn": ModuleSpec.create(standardize_libero_vggt),
        "image_obs_keys": {"primary": "image_primary"},
        "proprio_obs_key": "proprio",
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
    }

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
    window_size = FieldReference(default=2)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=32,
        shuffle_buffer_size=10000,
        num_steps=50000,
        log_interval=100,
        eval_interval=1000,
        save_interval=5000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="Geo_Octo", group=placeholder(str), entity="nlp-squad"
        ),
        
        dataset_kwargs_list=[DATASET_KWARGS],
        
        modality=task,
        finetuning_mode=mode,
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
    
    config['update_config'] = {"model": {"observation_tokenizers": {"vggt_tokens": ModuleSpec.create('octo.model.components.tokenizers:VGGTTokenizer')}}}


    return ConfigDict(config)