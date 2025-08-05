
from typing import Dict
import jax
import jax.numpy as jnp
from twoStwoR import TwoSTwoR


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

def gen_actions_set_policy(key) -> Dict[str, jax.Array]:
    """
    Generates a pre-defined set of actions for both Tree and Fungus agents.
    This is a placeholder function that can be replaced with a policy-based action generation.
    
    Returns:
        A dictionary with actions for both agents.
    """
    return {
        'tree': jnp.array([0.8, 0.2, 0.5, 0.2, 0.5, 0.1, 0.2]),
        'fungus': jnp.array([0., 1.0, 1.0, 0., 0.6, 0.2, 0.2])
    }

# Example simulation function for a single Tree agent in the TwoSTwoR environment
def run_simulation(config):
    """Runs a single simulation episode with randomised agent actions."""
    print("--- Initializing Environment and Agent ---")
    env = TwoSTwoR(grid_size=5)
    key = jax.random.key(config['SEED'])
    obs, state = env.reset(key)

    print(f"Initial Grid:\n{state.grid}\n")

    def env_step(runner_state, unused):
        """
        Steps the environment with the given actions, compatible with lax.scan.
        
        Args:
            env: The TwoSTwoR environment instance.
            key: JAX PRNG key for random number generation.
            state: Current state of the environment.
            actions: Actions to be performed by the agents.
        
        Returns:
            Updated state, rewards, and done flags.
        """
        state, key = runner_state
        key, action_key, step_key = jax.random.split(key, 3)

        # Actions selected randomly.
        actions = jax.lax.cond(
            config['GEN_ACTIONS'] == 'gen_random_actions',
            gen_random_actions,
            gen_actions_set_policy,
            action_key
        )

        # Step the environment
        obs, state, rewards, dones, shaped_rewards = env.step_env(step_key, state, actions)

        return (state, key), []

    (state, key), _ = jax.lax.scan(
        env_step,
        init=(state, key),
        xs=None,
        length=config['STEPS']
    )

    print("Final state after all steps:")

    print(f"Tree Biomass: {state.tree_agent.biomass:.2f}")
    print(f"Tree Health: {state.tree_agent.health:.2f}")
    print(f"Tree Sugars: {state.tree_agent.sugars:.2f}")
    print(f"Tree Phosphorus: {state.tree_agent.phosphorus:.2f}")
    print(f"Tree Defence: {state.tree_agent.defence:.2f}")
    print(f"Tree Radius: {state.tree_agent.radius:.2f}\n\n")

    print(f"Fungus Biomass: {state.fungus_agent.biomass:.2f}")
    print(f"Fungus Health: {state.fungus_agent.health:.2f}")
    print(f"Fungus Sugars: {state.fungus_agent.sugars:.2f}")
    print(f"Fungus Phosphorus: {state.fungus_agent.phosphorus:.2f}")
    print(f"Fungus Defence: {state.fungus_agent.defence:.2f}")
    print(f"Fungi Radius: {state.fungus_agent.radius:.2f}\n\n")

    print("--- Simulation Finished ---")

if __name__ == "__main__":
    config = {
        'STEPS': 100,
        'SEED': 0,
        'GEN_ACTIONS': 'gen_random_actions'  # or 'gen_actions_set_policy'
    }
    run_simulation(config)
