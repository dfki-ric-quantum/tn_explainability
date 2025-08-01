import jax
import jax.numpy as jnp


@jax.jit
def sv(s: jax.Array, v: jax.Array) -> jax.Array:
    """Compute SV/||SV||."""
    res = jnp.linalg.matmul(s, v)
    return res / jnp.linalg.norm(res)


@jax.jit
def us(u: jax.Array, s: jax.Array) -> jax.Array:
    """Compute US/||US||."""
    res = jnp.linalg.matmul(u, s)
    return res / jnp.linalg.norm(res)
