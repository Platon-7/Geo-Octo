from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec


def get_config():
    max_steps = FieldReference(10000)
    window_size = FieldReference(default=1)

    cfg = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        save_dir=placeholder(str),
        batch_size=8,
        shuffle_buffer_size=8192,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=1000,
        save_interval=2000,
        seed=42,
        dataset_kwargs=dict(
            name=placeholder(str),
            data_dir=placeholder(str),
            image_obs_keys={"primary": "image_0", "wrist": None},
            proprio_obs_key="proprio",
            language_key="language_instruction",
            action_proprio_normalization_type="normal",
            action_normalization_mask=[True, True, True, True, True, True, False],
            standardize_fn=ModuleSpec.create(
                "octo.data.oxe.oxe_standardization_transforms:bridge_dataset_transform",
            ),
        ),
        traj_transform_kwargs=dict(
            window_size=window_size,
            action_horizon=4,
            goal_relabeling_strategy="uniform",
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(keep_image_prob=1.0),
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (224, 224), "wrist": (128, 128)},
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
                wrist=dict(
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            ),
        ),
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
            frozen_keys=("octo_transformer.*",),
        ),
        model=dict(
            observation_tokenizers=dict(),
            task_tokenizers=dict(),
            heads=dict(
                action=ModuleSpec.create(
                    "octo.model.components.transformer:MAPHead",
                    num_readouts=1,
                    bottleneck_dim=1024,
                    output_dim=7,
                )
            ),
            readouts={"action": 1},
            transformer_kwargs=dict(
                num_layers=12,
                mlp_dim=2048,
                num_heads=8,
                dropout_rate=0.1,
                attention_dropout_rate=0.1,
            ),
            token_embedding_size=512,
            max_horizon=window_size,
            repeat_task_tokens=True,
            use_correct_attention=False,
            use_input_normalization=True,
            normalization_gate_scale=0.01,
        ),
    )

    return ConfigDict(cfg)
