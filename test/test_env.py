
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from twoStwoR import TwoSTwoR, P_AVAILABILITY, TREE_P_UPTAKE_EFFICIENCY, FUNGUS_P_UPTAKE_EFFICIENCY
from test_utils import gen_random_actions


def test_environment_initialization():
    env = TwoSTwoR(grid_size=5)
    assert env.grid_size == 5
    assert env.max_episode_steps == 100  # Assuming MAX_EPISODE_STEPS is 100

def test_environment_reset_state():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    assert state.grid.shape == (5, 5)
    assert state.step_count == 0

    assert state.tree_agent.position.shape == (2,)
    assert (state.tree_agent.position == jax.numpy.array([grid_size // 2, grid_size // 2])).all()  # Assuming tree starts at center (2, 2)
    assert state.tree_agent.species_id == 0  # Assuming species_id for Tree is 0
    assert state.tree_agent.health == 100.0  # Assuming initial health is 100
    assert state.tree_agent.biomass == .1  # Assuming initial biomass is 0.1
    assert state.tree_agent.phosphorus == 0.0  # Assuming initial phosphorus is 0
    assert state.tree_agent.sugars == 10.0  # Assuming initial sugars is 10
    assert state.tree_agent.defence == 1.0  # Assuming initial defence is 1.0
    assert state.tree_agent.radius == 0.0  # Assuming initial radius is 0

    assert state.fungus_agent.species_id == 1  # Assuming species_id for Fungus is 1
    assert state.fungus_agent.position.shape == (2,)
    assert (state.fungus_agent.position - jax.numpy.array([grid_size // 2, grid_size // 2]) <= 1).all()  # Fungus starts near the center, within 1 unit distance
    assert state.fungus_agent.health == 100.0  # Assuming initial health is 100
    assert state.fungus_agent.biomass == .1  # Assuming initial biomass is 0.1
    assert state.fungus_agent.phosphorus == 0.0  # Assuming initial phosphorus is 0
    assert state.fungus_agent.sugars == 10.0  # Assuming initial sugars is 10
    assert state.fungus_agent.defence == 1.0  # Assuming initial defence is 1.0
    assert state.fungus_agent.radius == 0.0  # Assuming initial radius is 0

def test_env_reset_jittable():
    env = TwoSTwoR()
    key = jax.random.PRNGKey(0)
    reset_fn = jax.jit(env.reset)
    obs, state = reset_fn(key)

    assert 'tree' in obs
    assert 'fungus' in obs

    tree_obs = obs['tree']
    fungus_obs = obs['fungus']

    assert tree_obs.shape == (6,)  # Position (2) + phosphorus (1) + sugars (1) + health (1)
    assert fungus_obs.shape == (6,)  # Same structure for fungus

def test_environment_observations():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    obs, _ = env.reset(key)

    assert 'tree' in obs
    assert 'fungus' in obs

    tree_obs = obs['tree']
    fungus_obs = obs['fungus']

    assert tree_obs.shape == (6,)  # Position (2) + phosphorus (1) + sugars (1) + health (1)
    # Check tree position is at the center of the grid
    assert (tree_obs[:2] == jax.numpy.array([grid_size // 2, grid_size // 2])).all()
    assert tree_obs[2] == 0.0  # Assuming initial phosphorus is 0
    assert tree_obs[3] == 10.0  # Assuming initial sugars is 10
    assert tree_obs[4] == 100.0  # Assuming initial health

    # Check fungus position is near grid center
    assert fungus_obs.shape == (6,)  # Same structure for fungus
    assert (fungus_obs[:2] - jax.numpy.array([grid_size // 2, grid_size // 2]) <= 1).all()
    assert fungus_obs[2] == 0.0  # Assuming initial phosphorus is 0
    assert fungus_obs[3] == 10.0  # Assuming initial sugars is 10
    assert fungus_obs[4] == 100.0  # Assuming initial health

def test_environment_observations_overlap():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)

    _, state = env.reset(key)

    state = jdc.replace(
        state,
        tree_agent=jdc.replace(state.tree_agent, radius=jnp.array(1.0)),
        fungus_agent=jdc.replace(state.fungus_agent, radius=jnp.array(1.0))
    ) # Start with zero radius by default, so set them to 1.0 to ensure overlap.

    obs = env._get_obs(state)  # Ensure the observation is generated

    tree_obs = obs['tree']
    fungus_obs = obs['fungus']

    if (tree_obs[:2] == fungus_obs[:2]).all():
        assert tree_obs[5] == 1 # Overlap with fungus
        assert fungus_obs[5] == 1  # Overlap with tree
    else:
        assert tree_obs[5] == 0 # No overlap with fungus
        assert fungus_obs[5] == 0 # No overlap with tree

def test_is_terminal_condition():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    # Check initial state is not terminal
    assert not env.is_terminal(state)

    # Simulate reaching max steps
    state = jdc.replace(state, step_count=env.max_episode_steps)
    assert env.is_terminal(state)

    # Simulate tree health reaching zero
    state = jdc.replace(state, tree_agent=jdc.replace(state.tree_agent, health=0.0))
    assert env.is_terminal(state)

    # Simulate fungus health reaching zero
    state = jdc.replace(state, fungus_agent=jdc.replace(state.fungus_agent, health=0.0))
    assert env.is_terminal(state)

    # Simulate tree occupying more than 25 grid cells
    state.grid.at[:, :].set(1)
    assert env.is_terminal(state)

    # Simulate fungus occupying more than 25 grid cells
    state.grid.at[:, :].set(2)
    assert env.is_terminal(state)

    # Simulate both occupying more than 25 grid cells
    state.grid.at[:, :].set(3)
    assert env.is_terminal(state)

def test_constrain_allocation_less_than_one():
    # Test with actions that sum to less than 1
    env = TwoSTwoR()

    actions = {
        'p_use': jnp.array(0.2),
        'p_trade': jnp.array(0.3),
        's_use': jnp.array(0.2),
        's_trade': jnp.array(0.2),
        'growth': jnp.array(0.1),
        'defence': jnp.array(0.1),
        'reproduction': jnp.array(0.1),
    }
    a = env.constrain_allocation(actions)
    s_allocation = [val for key,val in a.items() if 's_' in key]
    p_allocation = [val for key,val in a.items() if 's_' in key]
    s_use_allocation = [a['growth'], a['defence'], a['reproduction']]

    assert jax.numpy.sum(jnp.array(s_allocation)) <= 1.0
    assert jax.numpy.sum(jnp.array(p_allocation)) <= 1.0
    assert jax.numpy.sum(jnp.array(s_use_allocation)) <= 1.0

def test_constrain_allocation_equal_to_one():
    # Test with actions that sum to exactly 1
    env = TwoSTwoR()

    actions = {
        'p_use': jnp.array(0.3),
        'p_trade': jnp.array(0.7),
        's_use': jnp.array(0.5),
        's_trade': jnp.array(0.5),
        'growth': jnp.array(0.3),
        'defence': jnp.array(0.3),
        'reproduction': jnp.array(0.4),
    }
    a = env.constrain_allocation(actions)
    s_allocation = [val for key,val in a.items() if 's_' in key]
    p_allocation = [val for key,val in a.items() if 's_' in key]
    s_use_allocation = [a['growth'], a['defence'], a['reproduction']]

    assert jax.numpy.sum(jnp.array(s_allocation)) <= 1.0
    assert jax.numpy.sum(jnp.array(p_allocation)) <= 1.0
    assert jax.numpy.sum(jnp.array(s_use_allocation)) <= 1.0

    assert a['p_use'] == 0.3
    assert a['p_trade'] == 0.7
    assert a['s_use'] == 0.5
    assert a['s_trade'] == 0.5
    assert a['growth'] == 0.3
    assert a['defence'] == 0.3
    assert a['reproduction'] == 0.4

def test_constrain_allocation_greater_than_one():
    # Test with actions that sum to more than 1
    env = TwoSTwoR()

    actions = {
        'p_use': jnp.array(0.5),
        'p_trade': jnp.array(0.6),
        's_use': jnp.array(0.4),
        's_trade': jnp.array(0.9),
        'growth': jnp.array(0.3),
        'defence': jnp.array(0.4),
        'reproduction': jnp.array(0.5),
    }

    a = env.constrain_allocation(actions)
    s_allocation = [val for key,val in a.items() if 's_' in key]
    p_allocation = [val for key,val in a.items() if 's_' in key]
    s_use_allocation = [a['growth'], a['defence'], a['reproduction']]

    assert jax.numpy.sum(jnp.array(s_allocation)) <= 1.0
    assert jax.numpy.sum(jnp.array(p_allocation)) <= 1.0
    assert jax.numpy.sum(jnp.array(s_use_allocation)) <= 1.0

def test_constrain_allocation_all_zeroes():
    # Test with actions that sum to more than 1
    env = TwoSTwoR()

    actions = {
        'p_use': jnp.array(0.),
        'p_trade': jnp.array(0.),
        's_use': jnp.array(0.),
        's_trade': jnp.array(0.),
        'growth': jnp.array(0.),
        'defence': jnp.array(0.),
        'reproduction': jnp.array(0.),
    }

    a = env.constrain_allocation(actions)
    s_allocation = [val for key,val in a.items() if 's_' in key]
    p_allocation = [val for key,val in a.items() if 's_' in key]
    s_use_allocation = [a['growth'], a['defence'], a['reproduction']]

    assert jax.numpy.sum(jnp.array(s_allocation)) <= 1.0
    assert jax.numpy.sum(jnp.array(p_allocation)) <= 1.0
    assert jax.numpy.sum(jnp.array(s_use_allocation)) <= 1.0

def test_constrain_allocation_negative_values():
    # Test with actions that contain negative values
    # Ensure no negative values in the constrained actions
    env = TwoSTwoR()

    actions = {
        'p_use': jnp.array(-0.2),
        'p_trade': jnp.array(0.3),
        's_use': jnp.array(-0.1),
        's_trade': jnp.array(0.5),
        'growth': jnp.array(-0.4),
        'defence': jnp.array(0.2),
        'reproduction': jnp.array(0.3),
    }

    a = env.constrain_allocation(actions)

    assert [val for val in a.values() if val < 0] == []

def test_jittable_constrain_allocation():
    assert jax.jit(TwoSTwoR().constrain_allocation)

def test_allocate_resources_integer_values():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    key, action_key = jax.random.split(key)
    _, state = env.reset(key)

    actions_array = gen_random_actions(action_key)
    actions = jax.tree.map(lambda x: dict(zip(env.actions, x)), actions_array)

    tree_actions = env.allocate_resources(state.tree_agent, actions['tree'])
    
    # Used sugars isn't important after allocating resources, so we can ignore it
    tree_actions.pop('s_use', None)

    # Check all actions are integers
    vals = jnp.array(list(tree_actions.values()))
    assert jnp.allclose(vals // 1, vals)

def test_step_env_tree_resource_allocation_handled():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    # Manually set actions.
    actions_array = {
        'tree': jnp.array([1., 0., 0.5, 0.3, 0.4, 0.4, 0.2]),
        'fungus': jnp.array([0., 0.5, 1., 0., 0.4, 0.4, 0.2])
    }
    actions = jax.tree.map(lambda x: dict(zip(env.actions, x)), actions_array)

    # Initialize agents with some resources.
    sugars = 100.0
    f_phosphorus = 50.0
    t_phosphorus = 10.0

    state = jdc.replace(
        state, tree_agent=jdc.replace(
            state.tree_agent, phosphorus=jnp.array(t_phosphorus), sugars=jnp.array(sugars)
        )
    )
    state = jdc.replace(
        state, fungus_agent=jdc.replace(
            state.fungus_agent, phosphorus=jnp.array(f_phosphorus), sugars=jnp.array(sugars)
        )
    )


    _, new_state, _, _, _ = env.step_env(key, state, actions_array)

    # Check sugars generated based on phosphorus and canopy area
    A_c = 4.48  # Assuming biomass = 0.1
    I_s = 400.0  # Assume a fixed amount of sunlight available
    S_max = 1200. # Maximum sugar production rate
    K_I = 400. # Half-saturation constant for sugar production
    s_gen = jnp.floor(A_c * S_max * (I_s / (K_I + I_s)))
    p = 10  # Phosphorus used by tree

    sugars_generated = jnp.clip(s_gen, 0, p / 3)
    s_use = (actions['tree']['growth'] + actions['tree']['defence']) * actions['tree']['s_use'] # No seeds produced in this test.

    diff_t_sugars = - (s_use + actions['tree']['s_trade']) \
                    + actions['fungus']['s_trade'] # - 50. - 30. + 0.

    # Check phosphorus resources by tree.
    p_acquired = jnp.floor(A_c * P_AVAILABILITY * TREE_P_UPTAKE_EFFICIENCY)

    diff_t_p = - (actions['tree']['p_use'] + actions['tree']['p_trade']) * t_phosphorus \
               + (actions['fungus']['p_trade'] * f_phosphorus) # - 10. - 0. + 25.

    # Check expected resources generated/absorbed/traded by tree.
    assert np.allclose(new_state.tree_agent.sugars, state.tree_agent.sugars + (diff_t_sugars * sugars) + sugars_generated, atol=1e-5)
    assert np.allclose(new_state.tree_agent.phosphorus, state.tree_agent.phosphorus + diff_t_p + p_acquired, atol=1e-5)

def test_step_env_fungus_resource_allocation_handled():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    # Manually set actions.
    actions_array = {
        'tree': jnp.array([1., 0., 0.5, 0.3, 0.4, 0.4, 0.2]),
        'fungus': jnp.array([0., 0.5, 1., 0., 0.4, 0.4, 0.2])
    }
    actions = jax.tree.map(lambda x: dict(zip(env.actions, x)), actions_array)

    # Initialize agents with some resources.
    sugars = 100.0
    f_phosphorus = 50.0
    t_phosphorus = 10.0

    state = jdc.replace(
        state, tree_agent=jdc.replace(
            state.tree_agent, phosphorus=jnp.array(t_phosphorus), sugars=jnp.array(sugars)
        )
    )
    state = jdc.replace(
        state, fungus_agent=jdc.replace(
            state.fungus_agent, phosphorus=jnp.array(f_phosphorus), sugars=jnp.array(sugars)
        )
    )

    _, new_state, _, _, _ = env.step_env(key, state, actions_array)

    # Check sugars used/traded by fungus.
    s_use = actions['fungus']['growth'] + actions['fungus']['defence'] # 80; no seeds produced in this test.
    diff_f_sugars = - (s_use + actions['fungus']['s_trade']) + actions['tree']['s_trade'] # - 80. - 0. + 30.

    # Check phosphorus resources used/traded/absorbed by fungus.
    A_c = 4.47 * 1.5 # Scaling factor
    p_acquired = jnp.floor(A_c * P_AVAILABILITY * FUNGUS_P_UPTAKE_EFFICIENCY)

    diff_f_p = - (actions['fungus']['p_use'] + actions['fungus']['p_trade']) \
        + actions['tree']['p_trade']  # - 0. - 25. + 0..

    # Check expected resources generated/absorbed/traded by tree.
    assert new_state.fungus_agent.sugars ==  state.fungus_agent.sugars + (diff_f_sugars * sugars)
    assert new_state.fungus_agent.phosphorus == state.fungus_agent.phosphorus + (diff_f_p * f_phosphorus) + p_acquired

def test_step_env_jittable():
    env = TwoSTwoR()
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    actions = gen_random_actions(key)

    # Check if step_env is jittable
    step_env_jit = jax.jit(env.step_env)
    assert step_env_jit(key, state, actions) is not None

def test_step_env_state_update():
    grid_size = 5
    env = TwoSTwoR(grid_size=grid_size)
    key = jax.random.PRNGKey(42)
    _, state = env.reset(key)

    actions = gen_random_actions(key)

    _, new_state, _, _, _ = env.step_env(key, state, actions)

    assert new_state.step_count == 1  # Check step count increments

def test_step_env_dones():
    env = TwoSTwoR()
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    actions = gen_random_actions(key)

    _, _, _, dones, _ = env.step_env(key, state, actions)

    assert 'tree' in dones
    assert 'fungus' in dones
    assert '__all__' in dones

    assert not dones['tree']  # Initially, the environment should not be done
    assert not dones['fungus']  # Initially, the environment should not be done
    assert not dones['__all__']

def test_step_env_rewards():
    env = TwoSTwoR()
    key = jax.random.PRNGKey(42)
    _, state = env.reset(key)

    actions = gen_random_actions(key)

    _, _, rewards, _, _ = env.step_env(key, state, actions)

    assert 'tree' in rewards
    assert 'fungus' in rewards

    # Check that rewards are non-negative
    assert rewards['tree'] >= 0
    assert rewards['fungus'] >= 0

def test_step_env_shaped_rewards():
    env = TwoSTwoR()
    key = jax.random.PRNGKey(42)
    _, state = env.reset(key)

    actions = gen_random_actions(key)

    _, _, _, _, shaped_rewards = env.step_env(key, state, actions)

    assert 'tree' in shaped_rewards['shaped_reward']
    assert 'fungus' in shaped_rewards['shaped_reward']

    # # Check that shaped rewards are non-negative
    # ---- Currently outputting info for debugging and plotting – to be finalised later. ----
    # assert shaped_rewards['shaped_reward']['tree'] == {} # Not implemented yet
    # assert shaped_rewards['shaped_reward']['fungus'] == {} # Not implemented yet
