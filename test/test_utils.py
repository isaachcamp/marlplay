
from typing import Dict
import jax


def gen_random_actions(key) -> Dict[str, jax.Array]:
    """
    Generates random actions for both Tree and Fungus agents.
    Each action is a float in the range [0.0, 1.0].
    
    {'tree': jnp.array(7,), 'fungus': jnp.array(7,)}
    """
    key, tree_key, fungus_key = jax.random.split(key, 3)

    tree_actions = jax.random.uniform(tree_key, shape=(7,), minval=0.0, maxval=1.0)
    fungus_actions = jax.random.uniform(fungus_key, shape=(7,), minval=0.0, maxval=1.0)

    return {'tree': tree_actions, 'fungus': fungus_actions}
