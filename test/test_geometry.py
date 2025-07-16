
from twoStwoR.geometry import area_of_intersecting_circles
import jax.numpy as jnp
import jax

def test_area_of_intersecting_circles_no_intersection():
    xy1 = jnp.array([0.0, 0.0])
    r1 = jnp.array(1.0)
    xy2 = jnp.array([3.0, 0.0])
    r2 = jnp.array(1.0)
    area = area_of_intersecting_circles(xy1, r1, xy2, r2)
    assert jnp.isclose(area, 0.0)

def test_area_of_intersecting_circles_one_inside_another():
    xy1 = jnp.array([0.0, 0.0])
    r1 = jnp.array(3.0)
    xy2 = jnp.array([1.0, 0.0])
    r2 = jnp.array(1.0)
    area = area_of_intersecting_circles(xy1, r1, xy2, r2)
    assert jnp.isclose(area, jnp.pi * r2**2)

def test_area_of_intersecting_circles_partial_intersection():
    xy1 = jnp.array([0.0, 0.0])
    r1 = jnp.array(2.0)
    xy2 = jnp.array([1.5, 0.0])
    r2 = jnp.array(2.0)
    area = area_of_intersecting_circles(xy1, r1, xy2, r2)
    assert 0 < area < jnp.pi * r1**2

def test_area_of_intersecting_circles_jittable():
    assert jax.jit(area_of_intersecting_circles)
