
import jax
import jax.numpy as jnp


def area_of_intersecting_circles(
        xy1: jax.Array, r1: jax.Array,
        xy2: jax.Array, r2: jax.Array
    ) -> jax.Array:
    """
    Calculate the area of intersection between two circles, handling JAX JIT requirements.

    Parameters:
    xy1: tuple (x1, y1) - center of the first circle
    r1: float - radius of the first circle
    xy2: tuple (x2, y2) - center of the second circle
    r2: float - radius of the second circle

    Returns:
    float - area of intersection
    """
    def _area_of_intersecting_circles(d: jax.Array, r1: jax.Array, r2: jax.Array) -> jax.Array:
        """Calculate area of intersection between two circles."""
        a1 = r1**2 * jnp.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        a2 = r2**2 * jnp.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        area = a1 + a2 - 0.5 * jnp.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        return area

    def _area_encircled_circle(r1: jax.Array, r2: jax.Array) -> jax.Array:
        """Calculate area of smaller circle, assuming its within larger circle."""
        return jnp.pi * jnp.min(jnp.array([r1, r2])) ** 2

    d = jnp.linalg.norm(xy1 - xy2)

    return jax.lax.cond(
        d >= r1 + r2,
        lambda area: jnp.array(0.0), # No intersection
        lambda x: jax.lax.cond(
            x[0] <= abs(x[1] - x[2]),
            lambda x: _area_encircled_circle(x[1], x[2]), # One circle totally inside other
            lambda x: _area_of_intersecting_circles(*x),
            (d, r1, r2)
        ),
        (d, r1, r2)
    )
