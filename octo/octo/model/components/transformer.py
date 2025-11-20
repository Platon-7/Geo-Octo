# adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
from typing import Callable, Optional, Sequence, Union as _Union

import flax.linen as nn
from flax.linen import attention as attention_lib
import jax
import jax.numpy as jnp

from octo.model.components.base import TokenGroup
from octo.utils.typing import Dtype, PRNGKey, Shape, Union


class LoRADenseGeneral(nn.Module):
    """DenseGeneral layer with optional LoRA low-rank update.

    When lora_r > 0, the output becomes: y = DenseGeneral(x) + scale * (Dropout(x) @ A @ B),
    where A in R[in_features, r] and B in R[r, prod(features)].
    """

    features: _Union[int, Sequence[int]]
    axis: _Union[int, Sequence[int]] = -1
    use_bias: bool = True
    dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], jax.Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], jax.Array] = nn.initializers.normal(stddev=1e-6)

    # LoRA params
    lora_r: int = 0
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0

    @nn.compact
    def __call__(self, inputs: jax.Array, *, deterministic: bool) -> jax.Array:
        base = nn.DenseGeneral(
            features=self.features,
            axis=self.axis,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)

        # Fast path when LoRA disabled
        if self.lora_r is None or int(self.lora_r) <= 0:
            return base

        # Low-rank branch implemented as Dense(r) then DenseGeneral(features)
        x = inputs
        if self.lora_dropout and self.lora_dropout > 0.0:
            x = nn.Dropout(rate=self.lora_dropout)(x, deterministic=deterministic)

        down = nn.Dense(
            features=int(self.lora_r),
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.01),
            name="lora_A",
        )(x)
        up = nn.DenseGeneral(
            features=self.features,
            axis=self.axis,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
            name="lora_B",
        )(down)
        scale = self.lora_alpha / float(self.lora_r)
        delta = up * scale
        return base + delta.astype(base.dtype)


class LoRADense(nn.Module):
    """Dense layer with optional LoRA low-rank update."""

    features: int
    use_bias: bool = True
    dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], jax.Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], jax.Array] = nn.initializers.normal(stddev=1e-6)

    # LoRA
    lora_r: int = 0
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0

    @nn.compact
    def __call__(self, inputs: jax.Array, *, deterministic: bool) -> jax.Array:
        base = nn.Dense(
            features=self.features,
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)

        if self.lora_r is None or int(self.lora_r) <= 0:
            return base

        x = inputs
        if self.lora_dropout and self.lora_dropout > 0.0:
            x = nn.Dropout(rate=self.lora_dropout)(x, deterministic=deterministic)

        down = nn.Dense(
            features=int(self.lora_r),
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.01),
            name="lora_A",
        )(x)
        up = nn.Dense(
            features=self.features,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
            name="lora_B",
        )(down)
        delta = up * (self.lora_alpha / float(self.lora_r))
        return base + delta.astype(base.dtype)


class ResidualLoRA(nn.Module):
    """A low-rank residual adapter: delta = scale * up(dropout(down(x)))."""

    features: int
    dtype: Dtype = jnp.float32
    lora_r: int = 0
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        if self.lora_r is None or int(self.lora_r) <= 0:
            return jnp.zeros_like(x)
        z = x
        if self.lora_dropout and self.lora_dropout > 0.0:
            z = nn.Dropout(rate=self.lora_dropout)(z, deterministic=deterministic)
        z = nn.Dense(
            features=int(self.lora_r),
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.01),
            name="lora_down",
        )(z)
        z = nn.Dense(
            features=self.features,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
            name="lora_up",
        )(z)
        return z * (self.lora_alpha / float(self.lora_r))


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.

    Attributes:
      posemb_init: positional embedding initializer.
    """

    posemb_init: Callable[[PRNGKey, Shape, Dtype], jax.Array]

    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        )
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param("pos_embedding", self.posemb_init, pos_emb_shape)
        return inputs + pe


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[
        [PRNGKey, Shape, Dtype], jax.Array
    ] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], jax.Array] = nn.initializers.normal(
        stddev=1e-6
    )

    # LoRA
    use_lora: bool = False
    lora_r: int = 0
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="Dense_0",
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="Dense_1",
        )(x)
        if self.use_lora and self.lora_r and self.lora_r > 0:
            # Add residual low-rank adapters on each Dense output without changing base param names
            output = output + ResidualLoRA(
                features=actual_out_dim,
                dtype=self.dtype,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                name="lora_mlp_out",
            )(x, deterministic=deterministic)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output


class MAPHead(nn.Module):
    """Multihead Attention Pooling.

    From https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
    """

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 8
    num_readouts: int = 1

    @nn.compact
    def __call__(self, x: Union[jax.Array, TokenGroup], train=True):
        if isinstance(x, TokenGroup):
            x, mask = x.tokens, x.mask
        else:
            mask = None

        *batch_dims, l, d = x.shape
        x = x.reshape(-1, l, d)
        batch_size = x.shape[0]

        probe = self.param(
            "probe",
            nn.initializers.xavier_uniform(),
            (1, self.num_readouts, d),
            x.dtype,
        )
        probe = jnp.tile(probe, [batch_size, 1, 1])

        if mask is not None:
            mask = mask.reshape(-1, l)
            mask = jnp.broadcast_to(
                mask[:, None, None, :], (batch_size, 1, self.num_readouts, l)
            )

        out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform()
        )(probe, x, mask=mask)

        # TODO: dropout on head?
        y = nn.LayerNorm()(out)

        out = out + MlpBlock(mlp_dim=nn.merge_param("mlp_dim", self.mlp_dim, 4 * d))(
            y, deterministic=not train
        )
        out = out.reshape(*batch_dims, self.num_readouts, d)
        return out


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    # LoRA controls
    use_lora_attention: bool = False
    lora_r: int = 0
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0
    lora_attn_q: bool = True
    lora_attn_k: bool = False
    lora_attn_v: bool = True
    lora_attn_out: bool = False

    use_lora_mlp: bool = False
    capture_attention: bool = False

    @nn.compact
    def __call__(self, inputs, attention_mask, *, deterministic):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        # Base attention with original parameter names
        attn_layer = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_0",
        )
        if self.capture_attention:
            try:
                attn_weights = attention_lib.dot_product_attention_weights(
                    query,
                    key,
                    mask=attention_mask,
                    deterministic=True,
                    dtype=self.dtype,
                )
                self.sow("intermediates", "attention_weights", attn_weights)
            except Exception as exc:
                print(f"[ATTN] Failed to capture attention weights: {exc}", flush=True)
        attn_out = attn_layer(x, x, mask=attention_mask)
        # Optional LoRA on attention output projection only (safe w.r.t. pretrained names)
        if self.use_lora_attention and self.lora_r and self.lora_r > 0:
            delta_attn = ResidualLoRA(
                features=attn_out.shape[-1],
                dtype=self.dtype,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                name="lora_attn_out",
            )(x, deterministic=deterministic)
            attn_out = attn_out + delta_attn
        x = attn_out
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            use_lora=self.use_lora_mlp,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )(y, deterministic=deterministic)

        return x + y


class Transformer(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """

    num_layers: int
    mlp_dim: int
    num_attention_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    add_position_embedding: bool = False

    # Attention capture (for instrumentation)
    capture_attention_weights: bool = False

    # LoRA controls (forwarded to blocks)
    use_lora_attention: bool = False
    use_lora_mlp: bool = False
    lora_r: int = 0
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0
    lora_attn_q: bool = True
    lora_attn_k: bool = False
    lora_attn_v: bool = True
    lora_attn_out: bool = False

    @nn.compact
    def __call__(self, x, attention_mask, *, train):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert x.ndim == 3  # (batch, len, emb)

        if self.add_position_embedding:
            x = AddPositionEmbs(
                posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
                name="posembed_input",
            )(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Input Encoder
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoderblock_{lyr}",
                num_heads=self.num_attention_heads,
                # LoRA
                use_lora_attention=self.use_lora_attention,
                use_lora_mlp=self.use_lora_mlp,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_attn_q=self.lora_attn_q,
                lora_attn_k=self.lora_attn_k,
                lora_attn_v=self.lora_attn_v,
                lora_attn_out=self.lora_attn_out,
                capture_attention=self.capture_attention_weights,
            )(x, attention_mask, deterministic=not train)
        encoded = nn.LayerNorm(name="encoder_norm")(x)

        return encoded


def common_transformer_sizes(transformer_size: str) -> (int, dict):
    """
    Args:
        transformer_size (str): The size of the transformer. One of "dummy", "vanilla", "vit_s", "vit_b", "vit_l", "vit_h"

    Returns:
            token_embedding_size (int): The size of the token embeddings
            transformer_kwargs (dict): The kwargs to pass to the transformer

    """
    assert transformer_size in [
        "dummy",
        "vanilla",
        "vit_t",
        "vit_s",
        "vit_b",
        "vit_l",
        "vit_h",
    ]
    default_params = {
        "attention_dropout_rate": 0.0,
        "add_position_embedding": False,
    }

    TRANSFORMER_SIZES = {
        "dummy": dict(
            num_layers=1,
            mlp_dim=256,
            num_attention_heads=2,
            dropout_rate=0.1,
        ),
        "vanilla": dict(
            num_layers=4,
            mlp_dim=1024,
            num_attention_heads=8,
            dropout_rate=0.1,
        ),
        "vit_t": dict(
            num_layers=12,
            mlp_dim=768,
            num_attention_heads=3,
            dropout_rate=0.0,
        ),
        "vit_s": dict(
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.0,
        ),
        "vit_b": dict(
            num_layers=12,
            mlp_dim=3072,
            num_attention_heads=12,
            dropout_rate=0.0,
        ),
        "vit_l": dict(
            num_layers=24,
            mlp_dim=4096,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
        "vit_h": dict(
            num_layers=32,
            mlp_dim=5120,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
    }

    TOKEN_DIMS = {
        "dummy": 256,
        "vanilla": 256,
        "vit_t": 192,
        "vit_s": 384,
        "vit_b": 768,
        "vit_l": 1024,
        "vit_h": 1280,
    }

    return TOKEN_DIMS[transformer_size], {
        **default_params,
        **TRANSFORMER_SIZES[transformer_size],
    }