
from typing import Dict
import jax
import jax.numpy as jnp
from twoStwoR import TwoSTwoR


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

def gen_actions_set_policy(key) -> Dict[str, Dict[str, jax.Array]]:
    """
    Generates a pre-defined set of actions for both Tree and Fungus agents.
    This is a placeholder function that can be replaced with a policy-based action generation.
    
    Returns:
        A dictionary with actions for both agents.
    """
    return {
        'tree': {
            'p_use': jnp.array(0.8),
            'p_trade': jnp.array(0.2),
            's_use': jnp.array(0.5),
            's_trade': jnp.array(0.2),
            'growth': jnp.array(0.6),
            'defence': jnp.array(0.2),
            'reproduction': jnp.array(0.2)
        },
        'fungus': {
            'p_use': jnp.array(0.),
            'p_trade': jnp.array(1.0),
            's_use': jnp.array(1.0),
            's_trade': jnp.array(0.),
            'growth': jnp.array(0.6),
            'defence': jnp.array(0.2),
            'reproduction': jnp.array(0.2)
        }
    }

# Example simulation function for a single Tree agent in the TwoSTwoR environment
def run_simulation(seed: int = 42, num_steps: int = 50, gen_actions=gen_random_actions):
    """Runs a single simulation episode with randomised agent actions."""
    print("--- Initializing Environment and Agent ---")
    env = TwoSTwoR(grid_size=5)
    key = jax.random.PRNGKey(seed)
    _, state = env.reset(key)

    print(f"Initial Grid:\n{state.grid}\n")

    # --- Simulation Loop ---
    for i in range(num_steps):
        key, action_key, step_key = jax.random.split(key, 3)

        # Actions selected randomly.
        actions = gen_actions(action_key)

        # Step the environment
        _, state, rewards, dones, _ = env.step_env(step_key, state, actions)

        print(f"--- Step {i+1} ---\n\n")

        print(f"Tree Biomass: {state.tree_agent.biomass:.2f}")
        print(f"Tree Health: {state.tree_agent.health:.2f}")
        print(f"Tree Sugars: {state.tree_agent.sugars:.2f}")
        print(f"Tree Phosphorus: {state.tree_agent.phosphorus:.2f}")
        print(f"Tree Defence: {state.tree_agent.defence:.2f}\n")

        print(f"Fungus Biomass: {state.fungus_agent.biomass:.2f}")
        print(f"Fungus Health: {state.fungus_agent.health:.2f}")
        print(f"Fungus Sugars: {state.fungus_agent.sugars:.2f}")
        print(f"Fungus Phosphorus: {state.fungus_agent.phosphorus:.2f}")
        print(f"Fungus Defence: {state.fungus_agent.defence:.2f}\n\n")

        print("Tree Actions: ")
        print(f"Phosphorus used = {actions['tree']['p_use']:.2f},")
        print(f"Phosphorus traded = {actions['tree']['p_trade']:.2f},")
        print(f"Sugars used = {actions['tree']['s_use']:.2f},")
        print(f"Sugars traded = {actions['tree']['s_trade']:.2f},")
        print(f"Grow = {actions['tree']['growth']:.2f},")
        print(f"Defend = {actions['tree']['defence']:.2f},")
        print(f"Reproduce = {actions['tree']['reproduction']:.2f}")

        print(f"Reward: {rewards['tree']:.2f}\n")

        print("Fungus Actions: ")
        print(f"Phosphorus used={actions['fungus']['p_use']:.2f},")
        print(f"Phosphorus traded={actions['fungus']['p_trade']:.2f},")
        print(f"Sugars used={actions['fungus']['s_use']:.2f},")
        print(f"Sugars traded={actions['fungus']['s_trade']:.2f},")
        print(f"Grow={actions['fungus']['growth']:.2f},")
        print(f"Defend={actions['fungus']['defence']:.2f},")
        print(f"Reproduce={actions['fungus']['reproduction']:.2f}")

        print(f"Reward: {rewards['fungus']:.2f}\n")

        print(f"Current Grid:\n{state.grid}\n")

        if dones['__all__']:
            print(f"Episode finished at step {i+1}.")
            break

    print("Steps completed")
    print("--- Simulation Finished ---")

if __name__ == "__main__":
    run_simulation(seed=42, num_steps=100, gen_actions=gen_actions_set_policy)
