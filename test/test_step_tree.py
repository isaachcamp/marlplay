
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from twoStwoR import TwoSTwoR
from twoStwoR.env import (
    DEFENCE_CONSTANT, SUGARS_TO_BIOMASS, P_AVAILABILITY,
    PATHOGEN_ATTACK, SEED_COST, TREE_P_UPTAKE_EFFICIENCY,
)
from test_utils import gen_random_actions


def test_step_tree_defence():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    # Provide tree with sugars and allocate all to defence.
    state = jdc.replace(state,
        tree_agent=jdc.replace(state.tree_agent, sugars=jnp.array(10.0))
    )
    actions['tree']['s_use'] = actions['tree']['s_use'].at[()].set(1.0)
    actions['tree']['defence'] = actions['tree']['defence'].at[()].set(1.0)

    actions['tree']['growth'] = actions['tree']['growth'].at[()].set(0.0)
    actions['tree']['reproduction'] = actions['tree']['reproduction'].at[()].set(0.0)

    actions['tree'] = env.allocate_resources(state.tree_agent, actions['tree'])

    new_state, reward, _ = env.step_tree(key, state, actions['tree'])

    # Check tree health is reduced by defence cost
    assert new_state.tree_agent.defence == state.tree_agent.defence + (10.0 * DEFENCE_CONSTANT)
    assert new_state.tree_agent.sugars == 0. # Check sugars are used up.

    # Also tests growth and reproduction are unchanged as reward would be higher.
    assert reward == (state.tree_agent.defence + (10.0 * DEFENCE_CONSTANT)) * 0.1

def test_step_tree_growth():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    # Provide tree with sugars and allocate all to growth.
    state = jdc.replace(state,
        tree_agent=jdc.replace(state.tree_agent, sugars=jnp.array(10.0))
    )
    actions['tree']['s_use'] = actions['tree']['s_use'].at[()].set(1.0)
    actions['tree']['growth'] = actions['tree']['defence'].at[()].set(1.0)

    actions['tree']['defence'] = actions['tree']['growth'].at[()].set(0.0)
    actions['tree']['reproduction'] = actions['tree']['reproduction'].at[()].set(0.0)

    actions['tree'] = env.allocate_resources(state.tree_agent, actions['tree'])

    new_state, reward, _ = env.step_tree(key, state, actions['tree'])

    # Check tree health is reduced by growth cost
    assert new_state.tree_agent.health == 100.0 - (10.0 * PATHOGEN_ATTACK * 0.01)
    assert new_state.tree_agent.sugars == 0.

    expected_biomass_increase = 10.0 * SUGARS_TO_BIOMASS
    assert new_state.tree_agent.biomass == state.tree_agent.biomass + expected_biomass_increase

    # Also tests defence and reproduction are unchanged as reward would be higher.
    assert reward == expected_biomass_increase * 0.5 + 0.1 # +0.1 for base defence


def test_step_tree_reproduction():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    # Provide tree with sugars and allocate all to reproduction.
    state = jdc.replace(state,
        tree_agent=jdc.replace(state.tree_agent, sugars=jnp.array(100.0))
    )
    actions['tree']['s_use'] = actions['tree']['s_use'].at[()].set(1.0)
    actions['tree']['reproduction'] = actions['tree']['reproduction'].at[()].set(1.0)

    actions['tree']['growth'] = actions['tree']['growth'].at[()].set(0.0)
    actions['tree']['defence'] = actions['tree']['defence'].at[()].set(0.0)

    actions['tree'] = env.allocate_resources(state.tree_agent, actions['tree'])

    new_state, reward, _ = env.step_tree(key, state, actions['tree'])

    # Check number of seeds generated via rewards
    # Also tests defence and reproduction are unchanged as reward would be higher.
    no_seeds_generated = jnp.floor(100.0 / SEED_COST)
    assert reward == (no_seeds_generated * 1.5) + 0.1 # each seed worth 1.5 reward; +0.1 for base defence
    assert new_state.tree_agent.sugars == 0. # Check sugars are used up.

def test_step_tree_reproduction_non_integer_seeds():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    # Provide tree with sugars and allocate all to reproduction.
    state = jdc.replace(state,
        tree_agent=jdc.replace(state.tree_agent, sugars=jnp.array(120.0))
    )
    actions['tree']['s_use'] = actions['tree']['s_use'].at[()].set(1.0)
    actions['tree']['reproduction'] = actions['tree']['reproduction'].at[()].set(1.0)

    actions['tree']['s_trade'] = actions['tree']['s_trade'].at[()].set(0.0)
    actions['tree']['growth'] = actions['tree']['growth'].at[()].set(0.0)
    actions['tree']['defence'] = actions['tree']['defence'].at[()].set(0.0)

    actions['tree'] = env.allocate_resources(state.tree_agent, actions['tree'])

    new_state, reward, _ = env.step_tree(key, state, actions['tree'])

    # Check number of seeds generated via rewards
    no_seeds_generated = jnp.floor(120.0 / SEED_COST) # This should be 2.4, so 2 seeds.
    assert reward == (no_seeds_generated * 1.5) + 0.1 # each seed worth 1.5 reward; +0.1 for base defence
    assert new_state.tree_agent.sugars == 20. # Check some sugars remain.

def test_step_tree_phosphorus_absorption():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    state = jdc.replace(state,
        tree_agent=jdc.replace(state.tree_agent, biomass=jnp.array(1.0))
    )

    actions['tree'] = env.allocate_resources(state.tree_agent, actions['tree'])

    new_state, _, _ = env.step_tree(key, state, actions['tree'])

    # Check phosphorus is absorbed
    # Assumes initial biomass is 1., A_c ~ 10.16, p_uptake efficiency is 0.05
    assert new_state.tree_agent.phosphorus == jnp.floor(0.5 * P_AVAILABILITY)
    assert new_state.tree_agent.sugars == 10.  # No sugars generated with no initial P.

def test_step_tree_sugar_generation_low_phosphorus():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    # Provide tree with phosphorus and allocate all to sugar generation.
    p = 10.0  # set phosphorus to 10 for this test
    state = jdc.replace(state,
        tree_agent=jdc.replace(
            state.tree_agent,
            phosphorus=jnp.array(p),
            biomass=1.0  # Set biomass to 1 for canopy area calculation
        )
    )
    actions['tree']['p_use'] = actions['tree']['p_use'].at[()].set(1.0) # Use all phosphorus

    actions['tree'] = env.allocate_resources(state.tree_agent, actions['tree'])

    new_state, _, _ = env.step_tree(key, state, actions['tree'])

    # Check sugars generated based on phosphorus and canopy area
    A_c = 10.16  # Assuming biomass_to_canopy_area_allometry returns 1 for simplicity
    I_s = 400.0  # Assume a fixed amount of sunlight available
    S_max = 1200. # Maximum sugar production rate
    K_I = 400. # Half-saturation constant for sugar production
    s_gen = jnp.floor(A_c * S_max * (I_s / (K_I + I_s)))
    expected_sugars_generated = jnp.clip(s_gen, 0, p / 3)

    # Check expected resources generated/absorbed.
    assert new_state.tree_agent.sugars == expected_sugars_generated + state.tree_agent.sugars
    
    # Assumes initial biomass is 1., A_c ~ 10.16, p_uptake efficiency is 0.05
    assert new_state.tree_agent.phosphorus == jnp.floor(10.15 * TREE_P_UPTAKE_EFFICIENCY * P_AVAILABILITY)

def test_step_tree_sugar_generation_high_phosphorus():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions = gen_random_actions(action_key)

    # Provide tree with high phosphorus and allocate all to sugar generation.
    p = 10000.0
    state = jdc.replace(state,
        tree_agent=jdc.replace(
            state.tree_agent,
            phosphorus=jnp.array(p),
            biomass=1.0  # Set biomass to 1 for canopy area calculation
        )
    )
    actions['tree']['p_use'] = actions['tree']['p_use'].at[()].set(1.0) # Use all phosphorus

    actions['tree'] = env.allocate_resources(state.tree_agent, actions['tree'])

    new_state, _, _ = env.step_tree(key, state, actions['tree'])

    # Check sugars generated based on phosphorus and canopy area
    A_c = 10.16
    I_s = 400.0  # Assume a fixed amount of sunlight available
    S_max = 1200. # Maximum sugar production rate
    K_I = 400. # Half-saturation constant for sugar production
    s_gen = jnp.floor(A_c * S_max * (I_s / (K_I + I_s)))
    expected_sugars_generated = jnp.clip(s_gen, 0, p / 3)

    # Check expected resources generated/absorbed.
    assert new_state.tree_agent.sugars == expected_sugars_generated + state.tree_agent.sugars

    # Assumes initial biomass is 1., A_c ~ 10.16, p_uptake efficiency is 0.05
    assert new_state.tree_agent.phosphorus == jnp.floor(10.15 * TREE_P_UPTAKE_EFFICIENCY * P_AVAILABILITY)
