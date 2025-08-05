
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from twoStwoR import TwoSTwoR, TRADE_PER_UNIT_AREA
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
            state.tree_agent,
            phosphorus=jnp.array(20.0),
            sugars=jnp.array(50.0),
            radius=1.  # Set radius to 1.0 for contact area
        )
    )
    state = jdc.replace(
        state, fungus_agent=jdc.replace(
            state.fungus_agent,
            phosphorus=jnp.array(50.0), 
            sugars=jnp.array(10.0),
            radius=1.
        )
    )

    # Set up the grid with one cell contact for trade
    state.grid.at[grid_size // 2, grid_size // 2].set(3)  # Set contact area for trade

    # Generate random actions and map to action keys.
    actions = gen_random_actions(action_key)
    actions = jax.tree.map(lambda x: dict(zip(env.actions, x)), actions)

    actions['tree']['p_trade'] = actions['tree']['p_trade'].at[()].set(0.0)
    actions['tree']['s_trade'] = actions['tree']['s_trade'].at[()].set(1.0)
    actions['fungus']['p_trade'] = actions['fungus']['p_trade'].at[()].set(1.0)
    actions['fungus']['s_trade'] = actions['fungus']['s_trade'].at[()].set(0.0)

    actions['tree'] = env.allocate_resources(state.tree_agent, actions['tree'])
    actions['fungus'] = env.allocate_resources(state.fungus_agent, actions['fungus'])

    new_state = env.step_trade(state, actions)

    # Check trades are capped by TRADE_PER_UNIT_AREA
    contact_area = jnp.pi
    max_trade = contact_area * TRADE_PER_UNIT_AREA

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
            state.tree_agent,
            phosphorus=jnp.array(20.0),
            sugars=jnp.array(120.0),
            radius=.5,  # Set radius to .5 for contact area
        )
    )
    state = jdc.replace(
        state, fungus_agent=jdc.replace(
            state.fungus_agent,
            phosphorus=jnp.array(150.0),
            sugars=jnp.array(10.0),
            radius=.5,
        )
    )

    # Set up the grid with one cell contact for trade
    state.grid.at[grid_size // 2, grid_size // 2].set(3)  # Set contact area for trade

    # Generate random actions and map with action keys.
    actions = gen_random_actions(action_key)
    actions = jax.tree.map(lambda x: dict(zip(env.actions, x)), actions)

    actions['tree']['p_trade'] = actions['tree']['p_trade'].at[()].set(0.0)
    actions['tree']['s_trade'] = actions['tree']['s_trade'].at[()].set(1.0)
    actions['fungus']['p_trade'] = actions['fungus']['p_trade'].at[()].set(1.0)
    actions['fungus']['s_trade'] = actions['fungus']['s_trade'].at[()].set(0.0)

    actions['tree'] = env.allocate_resources(state.tree_agent, actions['tree'])
    actions['fungus'] = env.allocate_resources(state.fungus_agent, actions['fungus'])

    new_state = env.step_trade(state, actions)

    # Check trades are capped by TRADE_PER_UNIT_AREA
    contact_area = jnp.pi / 4
    max_trade = jnp.floor(contact_area * TRADE_PER_UNIT_AREA)

    print(f"Max trade: {max_trade}")

    assert (new_state.tree_agent.phosphorus - state.tree_agent.phosphorus) <= max_trade
    assert (new_state.tree_agent.sugars - state.tree_agent.sugars) <= max_trade
    assert (new_state.fungus_agent.phosphorus - state.fungus_agent.phosphorus) <= max_trade
    assert (new_state.fungus_agent.sugars - state.fungus_agent.sugars) <= max_trade

    assert new_state.tree_agent.phosphorus == 20.0 + max_trade
    assert new_state.tree_agent.sugars == 120.0 - max_trade
    assert new_state.fungus_agent.phosphorus == 150.0 - max_trade
    assert new_state.fungus_agent.sugars == 10.0 + max_trade
