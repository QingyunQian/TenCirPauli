"""Lazy JAX callback/VJP bridge for scalar circuit objectives."""

from __future__ import annotations

import importlib
from typing import Any, Iterable, Optional, cast

import numpy as np


def _require_jax() -> Any:
    try:
        jax = cast(Any, importlib.import_module("jax"))
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "expectation_jax() requires the optional 'jax' dependency"
        ) from error
    return jax


def native_expectation_jax(
    angles: Iterable[object],
    engine: Any,
    *,
    channel_values: Iterable[object] = (),
    checkpoint_interval: Optional[int] = None,
) -> Any:
    """Stage one native value-and-gradient callback with a custom VJP.

    Rotations and built-in channel scalars are separate logical JAX inputs but
    are concatenated into the one flat native scalar buffer at the callback
    boundary.
    """
    jax = _require_jax()
    jnp = jax.numpy
    use_x64 = bool(jax.config.read("jax_enable_x64"))
    jax_real_dtype = jnp.float64 if use_x64 else jnp.float32
    numpy_real_dtype = np.float64 if use_x64 else np.float32
    angle_values = tuple(angles)
    channel_parameter_values = tuple(channel_values)
    angle_count = len(angle_values)
    channel_count = len(channel_parameter_values)
    count = angle_count + channel_count
    if angle_count:
        values = jnp.stack(
            [jnp.asarray(angle, dtype=jax_real_dtype) for angle in angle_values], axis=0
        )
    else:
        values = jnp.zeros((0,), dtype=jax_real_dtype)
    if channel_count:
        channels = jnp.stack(
            [
                jnp.asarray(value, dtype=jax_real_dtype)
                for value in channel_parameter_values
            ],
            axis=0,
        )
    else:
        channels = jnp.zeros((0,), dtype=jax_real_dtype)
    value_spec = jax.ShapeDtypeStruct((), jax_real_dtype)
    gradient_spec = jax.ShapeDtypeStruct((count,), jax_real_dtype)

    def callback(
        runtime_angles: Any, runtime_channels: Any
    ) -> tuple[Any, np.ndarray[Any, Any]]:
        runtime_values = np.concatenate(
            (
                np.asarray(runtime_angles, dtype=numpy_real_dtype),
                np.asarray(runtime_channels, dtype=numpy_real_dtype),
            )
        )
        concrete = np.ascontiguousarray(runtime_values)
        if checkpoint_interval is None:
            result = engine.value_and_grad(concrete)
        else:
            result = engine.value_and_grad(
                concrete, checkpoint_interval=checkpoint_interval
            )
        gradient = np.ascontiguousarray(
            np.asarray(result.gradient, dtype=numpy_real_dtype).reshape((count,))
        )
        return numpy_real_dtype(result.value), gradient

    def call_native(runtime_angles: Any, runtime_channels: Any) -> tuple[Any, Any]:
        return cast(
            tuple[Any, Any],
            jax.pure_callback(
                callback,
                (value_spec, gradient_spec),
                runtime_angles,
                runtime_channels,
            ),
        )

    @jax.custom_vjp  # type: ignore[untyped-decorator]
    def native_scalar(runtime_angles: Any, runtime_channels: Any) -> Any:
        value, _ = call_native(runtime_angles, runtime_channels)
        return value

    def forward(runtime_angles: Any, runtime_channels: Any) -> tuple[Any, Any]:
        value, gradient = call_native(runtime_angles, runtime_channels)
        return value, gradient

    def backward(gradient: Any, cotangent: Any) -> tuple[Any, Any]:
        return (
            cotangent * gradient[:angle_count],
            cotangent * gradient[angle_count:],
        )

    native_scalar.defvjp(forward, backward)
    return native_scalar(values, channels)


__all__ = ["native_expectation_jax"]
