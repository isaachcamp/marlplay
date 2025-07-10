
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from twoStwoR import TwoSTwoR
from twoStwoR.env import TRADE_PER_CELL
from test_utils import gen_random_actions


def test_step_trade_less_than_max_trade():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    # Initialize agents with some resources
    state = jdc.replace(
        state, tree_agent=jdc.replace(
            state.tree_agent, phosphorus=jnp.array(20.0), sugars=jnp.array(50.0)
        )
    )
    state = jdc.replace(
        state, fungus_agent=jdc.replace(
            state.fungus_agent, phosphorus=jnp.array(50.0), sugars=jnp.array(10.0)
        )
    )

    # Set up the grid with one cell contact for trade
    state.grid.at[grid_size // 2, grid_size // 2].set(3)  # Set contact area for trade

    # Set up actions for tree and fungus agents
    actions = gen_random_actions(action_key)

    actions['tree']['p_trade'] = actions['tree']['p_trade'].at[()].set(0.0)
    actions['tree']['s_trade'] = actions['tree']['s_trade'].at[()].set(1.0)
    actions['fungus']['p_trade'] = actions['fungus']['p_trade'].at[()].set(1.0)
    actions['fungus']['s_trade'] = actions['fungus']['s_trade'].at[()].set(0.0)

    actions['tree'] = env.allocate_resources(state.tree_agent, actions['tree'])
    actions['fungus'] = env.allocate_resources(state.fungus_agent, actions['fungus'])

    new_state = env.step_trade(state, actions)

    # Check trades are capped by TRADE_PER_CELL
    contact_area = 1.
    max_trade = contact_area * TRADE_PER_CELL

    assert (new_state.tree_agent.phosphorus - state.tree_agent.phosphorus) <= max_trade
    assert (new_state.tree_agent.sugars - state.tree_agent.sugars) <= max_trade
    assert (new_state.fungus_agent.phosphorus - state.fungus_agent.phosphorus) <= max_trade
    assert (new_state.fungus_agent.sugars - state.fungus_agent.sugars) <= max_trade

    assert new_state.tree_agent.phosphorus == 70.0
    assert new_state.tree_agent.sugars == 0.0
    assert new_state.fungus_agent.phosphorus == 0.0
    assert new_state.fungus_agent.sugars == 60.0

def test_step_trade_exceeds_max_trade():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    # Initialize agents with some resources
    state = jdc.replace(
        state, tree_agent=jdc.replace(
            state.tree_agent, phosphorus=jnp.array(20.0), sugars=jnp.array(120.0)
        )
    )
    state = jdc.replace(
        state, fungus_agent=jdc.replace(
            state.fungus_agent, phosphorus=jnp.array(150.0), sugars=jnp.array(10.0)
        )
    )


    # Set up the grid with one cell contact for trade
    state.grid.at[grid_size // 2, grid_size // 2].set(3)  # Set contact area for trade

    # Set up actions for tree and fungus agents
    actions = gen_random_actions(action_key)

    actions['tree']['p_trade'] = actions['tree']['p_trade'].at[()].set(0.0)
    actions['tree']['s_trade'] = actions['tree']['s_trade'].at[()].set(1.0)
    actions['fungus']['p_trade'] = actions['fungus']['p_trade'].at[()].set(1.0)
    actions['fungus']['s_trade'] = actions['fungus']['s_trade'].at[()].set(0.0)

    actions['tree'] = env.allocate_resources(state.tree_agent, actions['tree'])
    actions['fungus'] = env.allocate_resources(state.fungus_agent, actions['fungus'])

    new_state = env.step_trade(state, actions)

    # Check trades are capped by TRADE_PER_CELL
    contact_area = 1.
    max_trade = contact_area * TRADE_PER_CELL

    assert (new_state.tree_agent.phosphorus - state.tree_agent.phosphorus) <= max_trade
    assert (new_state.tree_agent.sugars - state.tree_agent.sugars) <= max_trade
    assert (new_state.fungus_agent.phosphorus - state.fungus_agent.phosphorus) <= max_trade
    assert (new_state.fungus_agent.sugars - state.fungus_agent.sugars) <= max_trade

    assert new_state.tree_agent.phosphorus == 120.0
    assert new_state.tree_agent.sugars == 20.0
    assert new_state.fungus_agent.phosphorus == 50.0
    assert new_state.fungus_agent.sugars == 110.0
