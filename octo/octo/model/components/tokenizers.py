import logging
import re
from typing import Dict, Optional, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np

from octo.model.components.base import TokenGroup
from octo.utils.typing import Data 
from octo.model.components.transformer import MAPHead
from octo.utils.spec import ModuleSpec


EPS = 1e-6


def generate_proper_pad_mask(
    tokens: jax.Array,
    pad_mask_dict: Optional[Dict[str, jax.Array]],
    keys: Sequence[str],
) -> jax.Array:
    if pad_mask_dict is None:
        logging.warning("No pad_mask_dict found. Nothing will be masked.")
        return jnp.ones(tokens.shape[:-1])
    if not all([key in pad_mask_dict for key in keys]):
        logging.warning(
            f"pad_mask_dict missing keys {set(keys) - set(pad_mask_dict.keys())}."
            "Nothing will be masked."
        )
        return jnp.ones(tokens.shape[:-1])

    pad_mask = jnp.stack([pad_mask_dict[key] for key in keys], axis=-1)
    pad_mask = jnp.any(pad_mask, axis=-1)
    pad_mask = jnp.broadcast_to(pad_mask[..., None], tokens.shape[:-1])
    return pad_mask


class TokenLearner(nn.Module):
    """
    Learns to map fixed-length sequence of tokens into specified number of tokens.

    Args:
        num_tokens (int): Number of output tokens.
        bottleneck_dim (int): Size of the hidden layers of the mapping MLP.
        dropout_rate (float): Rate of dropout applied in the mapping MLP. Defaults to no dropout.
    """

    num_tokens: int

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (inputs.shape[-2], inputs.shape[-1]),
        )
        x = inputs + jnp.broadcast_to(pos_embed, inputs.shape)
        x = nn.LayerNorm()(x)
        return MAPHead(num_readouts=self.num_tokens)(x, train=train)


def regex_match(regex_keys, x):
    return any([re.match(r_key, x) for r_key in regex_keys])


def regex_filter(regex_keys, xs):
    return list(filter(lambda x: regex_match(regex_keys, x), xs))


class ImageTokenizer(nn.Module):
    """Image tokenizer that encodes image stack into tokens with optional FiLM conditioning.

    Args:
        encoder (ModuleSpec): Encoder class.
        use_token_learner (bool): Whether to use token learner. Defaults to False.
        num_tokens (int): Number of output tokens, only enforced when use_token_learner is True.
        obs_stack_keys (Sequence[str]): Which spatial observation inputs get stacked for encoder input. Supports regex.
        task_stack_keys (Sequence[str]): Which spatial task inputs get stacked for encoder input. Supports regex.
        task_film_keys (Sequence[str]): Which non-spatial task keys get passed into FiLM conditioning. Supports regex.
    """

    encoder: ModuleSpec
    use_token_learner: bool = False
    num_tokens: int = 8
    conditioning_type: str = "none"
    obs_stack_keys: Sequence[str] = ("image_.*", "depth_.*")
    task_stack_keys: Sequence[str] = tuple()
    task_film_keys: Sequence[str] = tuple()
    proper_pad_mask: bool = True

    @nn.compact
    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        def extract_inputs(keys, inputs, check_spatial=False):
            extracted_outputs = []
            for key in keys:
                if check_spatial:
                    assert len(inputs[key].shape) >= 4
                extracted_outputs.append(inputs[key])
            return jnp.concatenate(extracted_outputs, axis=-1)

        obs_stack_keys = regex_filter(self.obs_stack_keys, sorted(observations.keys()))
        if len(obs_stack_keys) == 0:
            logging.info(
                f"No image inputs matching {self.obs_stack_keys} were found."
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper_pad_mask."
            return None

        # stack all spatial observation and task inputs
        enc_inputs = extract_inputs(obs_stack_keys, observations, check_spatial=True)
        if self.task_stack_keys:
            needed_task_keys = regex_filter(self.task_stack_keys, observations.keys())
            # if any task inputs are missing, replace with zero padding (TODO: be more flexible)
            for k in needed_task_keys:
                if k not in tasks:
                    logging.info(
                        f"No task inputs matching {k} were found. Replacing with zero padding."
                    )
                    tasks = flax.core.copy(
                        tasks, {k: jnp.zeros_like(observations[k][:, 0])}
                    )
            task_stack_keys = regex_filter(self.task_stack_keys, sorted(tasks.keys()))
            if len(task_stack_keys) == 0:
                raise ValueError(
                    f"No task inputs matching {self.task_stack_keys} were found."
                )
            task_inputs = extract_inputs(task_stack_keys, tasks, check_spatial=True)
            task_inputs = task_inputs[:, None].repeat(enc_inputs.shape[1], axis=1)
            enc_inputs = jnp.concatenate([enc_inputs, task_inputs], axis=-1)
        b, t, h, w, c = enc_inputs.shape
        enc_inputs = jnp.reshape(enc_inputs, (b * t, h, w, c))

        # extract non-spatial FiLM inputs
        encoder_input_kwargs = {}
        if self.task_film_keys:
            film_inputs = extract_inputs(self.task_film_keys, tasks)
            film_inputs = film_inputs[:, None].repeat(t, axis=1)
            encoder_input_kwargs.update(
                {"cond_var": jnp.reshape(film_inputs, (b * t, -1))}
            )

        # run visual encoder
        encoder_def = ModuleSpec.instantiate(self.encoder)()
        image_tokens = encoder_def(enc_inputs, **encoder_input_kwargs)
        image_tokens = jnp.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))

        if self.use_token_learner:
            image_tokens = TokenLearner(num_tokens=self.num_tokens)(
                image_tokens, train=train
            )

        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                image_tokens,
                observations.get("pad_mask_dict", None),
                obs_stack_keys,
            )
        else:
            pad_mask = jnp.ones(image_tokens.shape[:-1])
        return TokenGroup(image_tokens, pad_mask)


class LanguageTokenizer(nn.Module):
    """
    Language tokenizer that embeds text input IDs into continuous language embeddings. Supports pre-trained HF models.

     Args:
         num_tokens (int): Number of output tokens (not enforced).
         encoder (str, optional): Optional HuggingFace AutoModel name for encoding input IDs.
         finetune_encoder (bool, optional): Optional finetune last layers of the language model.
    """

    encoder: str = None
    finetune_encoder: bool = False
    proper_pad_mask: bool = True

    def setup(self):
        if self.encoder is not None:
            from transformers import AutoConfig, FlaxAutoModel, FlaxT5EncoderModel

            config = AutoConfig.from_pretrained(self.encoder)
            if "t5" in self.encoder:
                self.hf_model = FlaxT5EncoderModel(config).module
            else:
                self.hf_model = FlaxAutoModel.from_config(config).module

    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        if "language_instruction" not in tasks:
            logging.warning("No language inputs found. Skipping tokenizer entirely.")
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
            return None

        if not isinstance(tasks["language_instruction"], (jax.Array, np.ndarray)):
            assert (
                self.encoder is not None
            ), "Received language tokens but no encoder specified."
            tokens = self.hf_model(**tasks["language_instruction"]).last_hidden_state
        else:
            # add a # tokens dimension to language
            if tasks["language_instruction"].ndim == 2:
                tokens = tasks["language_instruction"][:, None, :]
            else:
                tokens = tasks["language_instruction"]

        if not self.finetune_encoder:
            tokens = jax.lax.stop_gradient(tokens)

        # TODO: incorporate padding info from language tokens here too
        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                tokens,
                tasks.get("pad_mask_dict", None),
                ("language_instruction",),
            )
        else:
            pad_mask = jnp.ones(tokens.shape[:-1])

        return TokenGroup(tokens, pad_mask)


class BinTokenizer(nn.Module):
    """
    Tokenizes continuous inputs via dimension-wise binning in given range.

    Args:
        n_bins (int): Number of discrete bins per dimension.
        bin_type (str): Type of binning. ['uniform', 'normal' = Gaussian]
        low (float): Lower bound for bin range.
        high (float): Upper bound for bin range.
    """

    n_bins: int = 256
    bin_type: str = "uniform"
    low: float = 0
    high: float = 1

    def setup(self):
        if self.bin_type == "uniform":
            self.thresholds = jnp.linspace(self.low, self.high, self.n_bins + 1)
        elif self.bin_type == "normal":
            self.thresholds = norm.ppf(jnp.linspace(EPS, 1 - EPS, self.n_bins + 1))
        else:
            raise ValueError(
                f"Binning type {self.bin_type} not supported in BinTokenizer."
            )

    def __call__(self, inputs):
        if self.bin_type == "uniform":
            inputs = jnp.clip(inputs, self.low + EPS, self.high - EPS)
        inputs = inputs[..., None]
        token_one_hot = (inputs < self.thresholds[1:]) & (
            inputs >= self.thresholds[:-1]
        ).astype(jnp.uint8)
        output_tokens = jnp.argmax(token_one_hot, axis=-1)
        return output_tokens

    def decode(self, inputs):
        one_hot = jax.nn.one_hot(inputs, self.n_bins)
        bin_avgs = (self.thresholds[1:] + self.thresholds[:-1]) / 2
        outputs = jnp.sum(one_hot * bin_avgs, axis=-1)
        return outputs


class LowdimObsTokenizer(BinTokenizer):
    """
    Tokenizer for non-spatial observations. Optionally discretizes into bins per dimension (see BinTokenizer).

    Args:
        obs_keys (Sequence[str]): List of non-spatial keys to concatenate & tokenize. Supports regex.
        discretize (bool): If True, discretizes inputs per dimension, see BinTokenizer.
    """

    obs_keys: Sequence[str] = tuple()
    discretize: bool = False
    proper_pad_mask: bool = True

    def __call__(self, observations, *unused_args, **unused_kwargs):
        assert self.obs_keys, "Need to specify observation keys to tokenize."
        if len(regex_filter(self.obs_keys, sorted(observations.keys()))) == 0:
            logging.warning(
                f"No observation inputs matching {self.obs_keys} were found."
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
            return None

        tokenizer_inputs = []
        for o_key in self.obs_keys:
            for key in filter(re.compile(o_key).match, sorted(observations.keys())):
                assert (
                    len(observations[key].shape) == 3
                ), f"Only supports non-spatial inputs but {key} has shape {observations[key].shape}."
                tokenizer_inputs.append(observations[key])
        tokenizer_inputs = jnp.concatenate(tokenizer_inputs, axis=-1)
        if self.discretize:
            tokenized_inputs = super().__call__(tokenizer_inputs)
            tokens = jax.nn.one_hot(tokenized_inputs, self.n_bins)
        else:
            tokens = tokenizer_inputs[..., None]
        mask = jnp.ones(tokens.shape[:-1])
        return TokenGroup(tokens, mask)

### START MODIFICATION ###
class VGGTTokenizer(nn.Module):
    """
    A tokenizer for pre-computed VGGT tokens with memory optimization options.
    """
    use_compression: bool = True  # Enable compression to reduce memory
    compression_ratio: float = 0.5  # Compress to 50% of original size
    use_token_learner: bool = True  # Use token learner to reduce token count
    num_output_tokens: int = 128  # Reduce from 261 to 128 tokens
    use_gradient_checkpointing: bool = True  # Enable gradient checkpointing

    @nn.compact
    def __call__(
        self,
        observations: Data,
        tasks: Data,
        train: bool = False,
    ) -> TokenGroup:
        vggt_tokens = observations.get("vggt_tokens")
        if vggt_tokens is None:
            raise ValueError(
                "`vggt_tokens` not found in observations. Ensure they are added in the data pipeline."
            )
        
        # Convert to float32 if needed and optimize memory layout
        if vggt_tokens.dtype == jnp.float16:
            # Keep as float16 to save memory, but ensure proper handling
            tokens = vggt_tokens.astype(jnp.float16)
        else:
            tokens = vggt_tokens.astype(jnp.float32)
        
        # Optional compression: reduce feature dimension
        if self.use_compression:
            compressed_dim = int(tokens.shape[-1] * self.compression_ratio)
            tokens = nn.Dense(
                features=compressed_dim,
                name="vggt_compression"
            )(tokens)
        
        # Optional token reduction using TokenLearner
        if self.use_token_learner:
            # Add positional embeddings
            pos_embed = self.param(
                "pos_embed",
                nn.initializers.normal(stddev=0.02),
                (tokens.shape[-2], tokens.shape[-1]),
            )
            tokens = tokens + jnp.broadcast_to(pos_embed, tokens.shape)
            tokens = nn.LayerNorm(name="vggt_norm")(tokens)
            
            # Use TokenLearner to reduce token count
            tokens = TokenLearner(
                num_tokens=self.num_output_tokens,
            )(tokens, train=train)
        
        # Generate mask
        mask = jnp.ones(tokens.shape[:-1], dtype=jnp.bool_)
        
        return TokenGroup(tokens=tokens, mask=mask)
### END MODIFICATION ###


class VisionMixer(nn.Module):
    """
    A tokenizer that correctly handles a string-based configuration by using
    the ModuleSpec system to instantiate its sub-modules.
    """
    patch_tokenizer_spec: dict
    vggt_tokenizer_spec: dict

    def setup(self):
        """Initializes the sub-tokenizers from their string-based specifications."""

        # 1. Create a ModuleSpec object from the dictionary spec that uses strings.
        patch_spec_obj = ModuleSpec.create(
            self.patch_tokenizer_spec['module'],
            **self.patch_tokenizer_spec.get('kwargs', {})
        )
        # 2. Use the framework's instantiate helper to create the module.
        self.patch_tokenizer = ModuleSpec.instantiate(patch_spec_obj)()
        
        # Do the same for the VGGT tokenizer.
        vggt_spec_obj = ModuleSpec.create(
            self.vggt_tokenizer_spec['module'],
            **self.vggt_tokenizer_spec.get('kwargs', {})
        )
        self.vggt_tokenizer = ModuleSpec.instantiate(vggt_spec_obj)()
        
        # Define the projection layer.
        # Note: We need to access the 'num_features' from the original dictionary spec.
        target_embedding_dim = self.patch_tokenizer_spec['kwargs']['encoder']['kwargs']['num_features']
        self.vggt_projection = nn.Dense(features=target_embedding_dim)


    def __call__(
        self,
        observations: Data,
        tasks: Data,
        train: bool = False,
    ) -> TokenGroup:
        patch_tokens = self.patch_tokenizer(observations, tasks, train=train)
        vggt_tokens = self.vggt_tokenizer(observations, tasks, train=train)
        
        projected_vggt_tokens = self.vggt_projection(vggt_tokens.tokens)
        
        mixed_tokens = jnp.concatenate([patch_tokens.tokens, projected_vggt_tokens], axis=-2)
        mixed_mask = jnp.concatenate([patch_tokens.mask, vggt_tokens.mask], axis=-1)
        return TokenGroup(tokens=mixed_tokens, mask=mixed_mask)

### END MODIFICATION ###