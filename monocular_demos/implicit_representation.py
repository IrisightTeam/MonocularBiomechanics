from typing import Any, Literal, Optional, List, Callable, Union
from jaxtyping import Array, PRNGKeyArray
import jax
import jax.numpy as jnp
import equinox as eqx

import numpy as np

import jmp
import math

import jax.random as jrandom

jmp_default_policy = jmp.get_policy("params=float32,compute=float16,output=float32")

non_linearity_lookup = {
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jnp.tanh,
    "softplus": jax.nn.softplus,
    "elu": jax.nn.elu,
    "gelu": jax.nn.gelu,
    "sin": jnp.sin,
    "mish": jax.nn.mish,
    "leaky_relu": jax.nn.leaky_relu,
    "selu": jax.nn.selu,
}


class Linear(eqx.Module, strict=True):
    """Performs a linear transformation."""

    weight: Array
    bias: Optional[Array]
    in_features: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_features: Union[int, Literal["scalar"]] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    jmp_policy: jmp.Policy = eqx.field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        custom_bias: Optional[List[float]] = None,
        jmp_policy: jmp.Policy = jmp_default_policy,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `in_features`: The input size. The input to the layer should be a vector of
            shape `(in_features,)`
        - `out_features`: The output size. The output from the layer will be a vector
            of shape `(out_features,)`.
        - `use_bias`: Whether to add on a bias as well.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `custom_bias`: A custom bias for the output. Should be the same length
            as the `out_features` parameter. Should be nan to not replace that bias
            element and the value to replace it with otherwise.

        Note that `in_features` also supports the string `"scalar"` as a special value.
        In this case the input to the layer should be of shape `()`.

        Likewise `out_features` can also be a string `"scalar"`, in which case the
        output from the layer will have shape `()`.
        """
        wkey, bkey = jrandom.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1 / math.sqrt(in_features_)

        dtype = jmp_policy.param_dtype

        self.weight = jrandom.uniform(wkey, (out_features_, in_features_), minval=-lim, maxval=lim, dtype=dtype)
        if use_bias:
            self.bias = jrandom.uniform(bkey, (out_features_,), minval=-lim, maxval=lim, dtype=dtype)

            if custom_bias is not None:
                self.bias = jnp.where(jnp.isnan(custom_bias), self.bias, custom_bias)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.jmp_policy = jmp_policy

    @jax.named_scope("eqx.nn.Linear")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_features,)`. (Or shape
            `()` if `in_features="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        !!! info

            If you want to use higher order tensors as inputs (for example featuring "
            "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
            "shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.

        **Returns:**

        A JAX array of shape `(out_features,)`. (Or shape `()` if
        `out_features="scalar"`.)
        """

        weight, bias, x = self.jmp_policy.cast_to_compute((self.weight, self.bias, x))

        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = weight @ x
        if bias is not None:
            x = x + bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x


def calculate_encoding_length(max_time: float, min_freq: float = 80):
    """
    Calculate the encoding length needed for a given maximum time and minimum frequency.

    Args:
        max_time: maximum time of the trajectory in seconds
        min_freq: minimum frequency of the trajectory in Hz

    Returns
        encoding_length: the number of encoding dimensions needed
    """
    base_freq_needed = min_freq * max_time / (2 * np.pi)
    encoding_length = jnp.ceil(jnp.log2(base_freq_needed) + 1).astype(int)
    return int(encoding_length)


def positional_encoding(inputs, positional_encoding_dims=3):
    batch_size, _ = inputs.shape
    inputs_freq = jax.vmap(lambda x: inputs * 2.0**x)(jnp.arange(positional_encoding_dims))
    periodic_fns = jnp.stack([jnp.sin(inputs_freq), jnp.cos(inputs_freq)])
    periodic_fns = periodic_fns.swapaxes(0, 2).reshape([batch_size, -1])
    periodic_fns = jnp.concatenate([inputs, periodic_fns], axis=-1)
    return periodic_fns


class ImplicitTrajectory(eqx.Module):
    layers: List[eqx.nn.Linear]
    final: eqx.Module
    max_time: float = jnp.array
    joints: int = eqx.field(static=True)
    spatial_dims: int = eqx.field(static=True)
    concatenate_layers: int = eqx.field(static=True)
    encoding_length: int = eqx.field(static=True)
    layer_norm: bool = eqx.field(static=True)
    jmp_policy: jmp.Policy = eqx.field(static=True)
    non_linearity: Callable = eqx.field(static=True)

    def __init__(
        self,
        features=[128, 256, 512, 1024, 2048, 2048, 4096],
        joints=75,
        spatial_dims=3,
        concatenate_layers=3,
        encoding_length=8,
        max_time=60,
        layer_norm=False,
        custom_bias=None,
        jmp_policy: jmp.Policy = jmp_default_policy,
        non_linearity: Callable = jax.nn.relu,
        *,
        key,  # 4,
    ):
        # self.features = features
        self.joints = joints
        self.concatenate_layers = 4
        self.spatial_dims = spatial_dims
        self.concatenate_layers = concatenate_layers
        self.encoding_length = encoding_length
        self.max_time = jnp.array(jnp.ceil(max_time), dtype=jnp.float32)
        self.layer_norm = layer_norm

        if isinstance(jmp_policy, str):
            jmp_policy = jmp.get_policy(jmp_policy)
        self.jmp_policy = jmp_policy

        # make sure this is not a jax datatype or it will trigger
        # tracing  errors
        encoding_size = int(self.encoding_length * 2 + 1)

        keys = jax.random.split(key, len(features) + 1)
        self.layers = []
        # self.ln_layers = []

        if self.concatenate_layers > len(features):
            raise ValueError("concatenate_layers must be less than or equal to the number of layers")

        for i, feat in enumerate(features):
            input_dim = encoding_size if i == 0 else features[i - 1]
            if i - 1 < self.concatenate_layers and i > 0:
                input_dim += encoding_size
            layer = Linear(in_features=input_dim, out_features=feat, key=keys[i], jmp_policy=jmp_policy)
            self.layers.append(layer)
            # layer = eqx.nn.LayerNorm(shape=feat, use_weight=False, use_bias=False)
            # self.ln_layers.append(layer)

        self.final = Linear(
            in_features=features[-1], out_features=self.joints * self.spatial_dims, key=keys[-1], jmp_policy=jmp_policy, custom_bias=custom_bias
        )

        if isinstance(non_linearity, str):
            non_linearity = non_linearity_lookup[non_linearity]

        self.non_linearity = non_linearity

    def __call__(self, time_point):
        time_point = self.jmp_policy.cast_to_param(time_point) # do positional encoding with high precision

        normalized_time_point = time_point / self.max_time * jnp.pi # normalize to [0, pi] so positional encoding does not alias
        encoded_time_point = positional_encoding(normalized_time_point.reshape((1, 1)), self.encoding_length)[0]
        encoded_time_point = self.jmp_policy.cast_to_compute(encoded_time_point) # now do the rest of the computation with lower precision

        x = encoded_time_point

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norm:
                x = eqx.nn.LayerNorm(shape=x.shape, use_weight=False, use_bias=False)(x)
            x = self.non_linearity(x)
            # x = self.ln_layers[i](x)
            if i < self.concatenate_layers:
                x = jnp.concatenate([x, encoded_time_point], axis=-1)
        x = self.final(x)
        x = x.reshape(self.joints, self.spatial_dims)

        x = self.jmp_policy.cast_to_output(x)

        return x


class ExplicitTrajectory(eqx.Module):
    pose_params: jnp.array

    def __init__(self, sequence_length, joints=75, spatial_dims=3):
        # self.pose_params = jnp.zeros((sequence_length, joints, spatial_dims))
        self.pose_params = jnp.array(np.random.normal(size=(sequence_length, joints, spatial_dims))) * 0.01

        # self.pose_params = self.pose_params + jnp.arange(sequence_length).reshape((sequence_length, 1, 1))

    def __call__(self, time_point):
        # interpolate into the sequence, which is assumed to have a time axis corresponding
        # to (0, 1)

        # return jnp.interp(time_point, jnp.linspace(0, 1, self.pose_params.shape[0]), self.pose_params)

        idx = jnp.linspace(0, 1, self.pose_params.shape[0])
        # find the nearest index and take from the sequence
        idx = jnp.argmin(jnp.abs(idx - time_point))
        return self.pose_params[idx]


class CubicSplineTrajectory(eqx.Module):
    """
    CubicSplineTrajectory implements a cubic (Catmull–Rom) spline interpolation
    over a set of control points. It is drop-in compatible with ImplicitTrajectory,
    producing an output of shape (joints, spatial_dims) given a scalar time input.

    Attributes:
        control_points: A trainable array of shape (sequence_length, joints, spatial_dims)
            representing the keyframes of the trajectory.
        max_time: The maximum time (in seconds) corresponding to the end of the trajectory.
        sequence_length: Number of control points.
        joints: Number of joints.
        spatial_dims: Dimensionality of each joint (typically 3).
    """

    control_points: Array  # shape: (sequence_length, joints, spatial_dims)
    max_time: float = jnp.array
    sequence_length: int = eqx.field(static=True)
    joints: int = eqx.field(static=True)
    spatial_dims: int = eqx.field(static=True)

    def __init__(
        self,
        sequence_length: int,
        joints: int = 75,
        spatial_dims: int = 3,
        max_time: float = 60.0,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize the CubicSplineTrajectory with random control points.

        Args:
            sequence_length: Number of keyframes/control points.
            joints: Number of joints.
            spatial_dims: Spatial dimensionality per joint.
            max_time: Maximum time (seconds) corresponding to the last control point.
            key: PRNG key for initializing the control points.
        """
        self.sequence_length = sequence_length
        self.joints = joints
        self.spatial_dims = spatial_dims
        self.max_time = max_time
        # Initialize control points with small random values.
        self.control_points = jrandom.normal(key, (sequence_length, joints, spatial_dims)) * 0.01

    def __call__(self, time_point: Array) -> Array:
        """
        Given a time point, returns the interpolated pose (of shape (joints, spatial_dims))
        using Catmull–Rom spline interpolation over the control points.

        Args:
            time_point: A scalar or 0-D array representing the current time.

        Returns:
            A JAX array of shape (joints, spatial_dims) representing the interpolated pose.
        """
        # Normalize time to [0, 1]
        normalized_time = jnp.clip(time_point / jax.lax.stop_gradient(self.max_time), 0.0, 1.0)
        # Scale time to the control point index range.
        scaled_t = normalized_time * (self.sequence_length - 1)

        i = jnp.floor(scaled_t).astype(jnp.int32)
        s = scaled_t - i  # local interpolation parameter in [0,1]

        # Get indices for the four control points needed (clamped to valid range).
        idx0 = jnp.clip(i - 1, 0, self.sequence_length - 1)
        idx1 = jnp.clip(i, 0, self.sequence_length - 1)
        idx2 = jnp.clip(i + 1, 0, self.sequence_length - 1)
        idx3 = jnp.clip(i + 2, 0, self.sequence_length - 1)

        p0 = self.control_points[idx0]
        p1 = self.control_points[idx1]
        p2 = self.control_points[idx2]
        p3 = self.control_points[idx3]

        # Compute the Catmull-Rom spline interpolation.
        s2 = s * s
        s3 = s2 * s
        pose = 0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2 + (-p0 + 3 * p1 - 3 * p2 + p3) * s3)
        return pose
