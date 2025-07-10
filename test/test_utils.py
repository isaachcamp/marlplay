
from typing import Dict
import jax


def gen_random_actions(key) -> Dict[str, Dict[str, jax.Array]]:
    """
    Generates random actions for both Tree and Fungus agents.
    Each action is a float in the range [0.0, 1.0].
    
    {'tree': {'p_use': float, ' p_trade': float, 's_use': float, 's_trade': float, 
              'growth': float, 'defence': float, 'reproduction': float},
    'fungus': {'p_use': float, ' p_trade': float, 's_use': float, 's_trade': float, 
               'growth': float, 'defence': float, 'reproduction': float}}
    """
    key, tree_key, fungus_key = jax.random.split(key, 3)

    tree_actions = {
        'p_use': jax.random.uniform(tree_key, shape=(), minval=0.0, maxval=1.0),
        'p_trade': jax.random.uniform(tree_key, shape=(), minval=0.0, maxval=1.0),
        's_use': jax.random.uniform(tree_key, shape=(), minval=0.0, maxval=1.0),
        's_trade': jax.random.uniform(tree_key, shape=(), minval=0.0, maxval=1.0),
        'growth': jax.random.uniform(tree_key, shape=(), minval=0.0, maxval=1.0),
        'defence': jax.random.uniform(tree_key, shape=(), minval=0.0, maxval=1.0),
        'reproduction': jax.random.uniform(tree_key, shape=(), minval=0.0, maxval=1.0)
    }
    fungus_actions = {
        'p_use': jax.random.uniform(fungus_key, shape=(), minval=0.0, maxval=1.0),
        'p_trade': jax.random.uniform(fungus_key, shape=(), minval=0.0, maxval=1.0),
        's_use': jax.random.uniform(fungus_key, shape=(), minval=0.0, maxval=1.0),
        's_trade': jax.random.uniform(fungus_key, shape=(), minval=0.0, maxval=1.0),
        'growth': jax.random.uniform(fungus_key, shape=(), minval=0.0, maxval=1.0),
        'defence': jax.random.uniform(fungus_key, shape=(), minval=0.0, maxval=1.0),
        'reproduction': jax.random.uniform(fungus_key, shape=(), minval=0.0, maxval=1.0)
    }

    return {'tree': tree_actions, 'fungus': fungus_actions}
