
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from twoStwoR import TwoSTwoR
from twoStwoR.env import (
    DEFENCE_CONSTANT, SUGARS_TO_BIOMASS, P_AVAILABILITY,
    PATHOGEN_ATTACK, SEED_COST
)
from test_utils import gen_random_actions


def test_step_fungus_defence():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    # Provide fungus with sugars and allocate all to defence.
    state = jdc.replace(state,
        fungus_agent=jdc.replace(state.fungus_agent, sugars=jnp.array(10.0))
    )
    actions['fungus']['s_use'] = actions['fungus']['s_use'].at[()].set(1.0)
    actions['fungus']['defence'] = actions['fungus']['defence'].at[()].set(1.0)

    actions['fungus']['growth'] = actions['fungus']['growth'].at[()].set(0.0)
    actions['fungus']['reproduction'] = actions['fungus']['reproduction'].at[()].set(0.0)

    actions['fungus'] = env.allocate_resources(state.fungus_agent, actions['fungus'])

    new_state, reward, _ = env.step_fungus(key, state, actions['fungus'])

    # Check fungus health is reduced by defence cost
    assert new_state.fungus_agent.defence == state.fungus_agent.defence + (10.0 * DEFENCE_CONSTANT)
    assert new_state.fungus_agent.sugars == 0. # Check sugars are used up.

    # Also tests growth and reproduction are unchanged as reward would be higher.
    assert reward == (state.fungus_agent.defence + (10.0 * DEFENCE_CONSTANT)) * 0.1

def test_step_fungus_growth():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    # Provide fungus with sugars and allocate all to growth.
    state = jdc.replace(state,
        fungus_agent=jdc.replace(state.fungus_agent, sugars=jnp.array(10.0))
    )
    actions['fungus']['s_use'] = actions['fungus']['s_use'].at[()].set(1.0)
    actions['fungus']['growth'] = actions['fungus']['defence'].at[()].set(1.0)

    actions['fungus']['defence'] = actions['fungus']['growth'].at[()].set(0.0)
    actions['fungus']['reproduction'] = actions['fungus']['reproduction'].at[()].set(0.0)

    actions['fungus'] = env.allocate_resources(state.fungus_agent, actions['fungus'])

    new_state, reward, _ = env.step_fungus(key, state, actions['fungus'])

    # Check fungus health is reduced by growth cost
    assert new_state.fungus_agent.health == 100.0 - (state.fungus_agent.biomass * PATHOGEN_ATTACK * 0.01)
    assert new_state.fungus_agent.sugars == 0.

    expected_biomass_increase = 10.0 * SUGARS_TO_BIOMASS
    assert new_state.fungus_agent.biomass == state.fungus_agent.biomass + expected_biomass_increase

    # Also tests defence and reproduction are unchanged as reward would be higher.
    assert reward == expected_biomass_increase * 0.5 + 0.1 # +0.1 for base defence


def test_step_fungus_reproduction():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    # Provide fungus with sugars and allocate all to reproduction.
    state = jdc.replace(state,
        fungus_agent=jdc.replace(state.fungus_agent, sugars=jnp.array(100.0))
    )
    actions['fungus']['s_use'] = actions['fungus']['s_use'].at[()].set(1.0)
    actions['fungus']['reproduction'] = actions['fungus']['reproduction'].at[()].set(1.0)

    actions['fungus']['growth'] = actions['fungus']['growth'].at[()].set(0.0)
    actions['fungus']['defence'] = actions['fungus']['defence'].at[()].set(0.0)

    actions['fungus'] = env.allocate_resources(state.fungus_agent, actions['fungus'])

    new_state, reward, _ = env.step_fungus(key, state, actions['fungus'])

    # Check number of seeds generated via rewards
    # Also tests defence and reproduction are unchanged as reward would be higher.
    no_seeds_generated = jnp.floor(100.0 / SEED_COST)
    assert reward == (no_seeds_generated * 1.5) + 0.1 # each seed worth 1.5 reward; +0.1 for base defence
    assert new_state.fungus_agent.sugars == 0. # Check sugars are used up.

def test_step_fungus_reproduction_non_integer_seeds():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    # Provide fungus with sugars and allocate all to reproduction.
    state = jdc.replace(state,
        fungus_agent=jdc.replace(state.fungus_agent, sugars=jnp.array(120.0))
    )
    actions['fungus']['s_use'] = actions['fungus']['s_use'].at[()].set(1.0)
    actions['fungus']['reproduction'] = actions['fungus']['reproduction'].at[()].set(1.0)

    actions['fungus']['s_trade'] = actions['fungus']['s_trade'].at[()].set(0.0)
    actions['fungus']['growth'] = actions['fungus']['growth'].at[()].set(0.0)
    actions['fungus']['defence'] = actions['fungus']['defence'].at[()].set(0.0)

    actions['fungus'] = env.allocate_resources(state.fungus_agent, actions['fungus'])

    new_state, reward, _ = env.step_fungus(key, state, actions['fungus'])

    # Check number of seeds generated via rewards
    no_seeds_generated = jnp.floor(120.0 / SEED_COST) # This should be 2.4, so 2 seeds.
    assert reward == (no_seeds_generated * 1.5) + 0.1 # each seed worth 1.5 reward; +0.1 for base defence
    assert new_state.fungus_agent.sugars == 20. # Check some sugars remain.

def test_step_fungus_phosphorus_absorption():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    state = jdc.replace(state,
        fungus_agent=jdc.replace(state.fungus_agent, biomass=jnp.array(1.0))
    )

    actions['fungus']['s_use'] = actions['fungus']['s_use'].at[()].set(0.0)

    actions['fungus'] = env.allocate_resources(state.fungus_agent, actions['fungus'])

    new_state, _, _ = env.step_fungus(key, state, actions['fungus'])

    # Check phosphorus is absorbed
    # Assumes initial biomass is 1., A_c ~ 15.23, p_uptake efficiency is 1.
    assert new_state.fungus_agent.phosphorus == 15.2 * P_AVAILABILITY # accounts for jnp.floor operation
    assert new_state.fungus_agent.sugars == 10.  # No sugars used, no generation.
