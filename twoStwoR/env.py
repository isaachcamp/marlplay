
from typing import Dict

import jax
from jax import lax
import jax.numpy as jnp
import jax_dataclasses as jdc

from twoStwoR.geometry import area_of_intersecting_circles


# Constants for the grid world
GRID_SIZE = 10
MAX_EPISODE_STEPS = 100

# Constants for the agents
SEED_COST = 50.0          # Energy cost per seed or spore
SUGARS_TO_BIOMASS = 0.005 # Conversion rate of sugars to biomass, assumes both tree and fungus have the same conversion rate and metabolic efficiency.
P_AVAILABILITY = 5.0      # Phosphorus availability in the environment
DEFENCE_CONSTANT = .05    # Scaling factor for defence increase
PATHOGEN_ATTACK = 1.      # Attack strength of pathogens on agent health
TRADE_PER_UNIT_AREA = 100      # Maximum amount of resources that can be traded per contact cell

TREE_P_UPTAKE_EFFICIENCY = .05 # Fixed efficiency for P absorption of tree.
FUNGUS_P_UPTAKE_EFFICIENCY = 1. # Fixed efficiency for P absorption of fungus.

# Do I need shaped rewards?
# BASE_REW_SHAPING_PARAMS = {
#     "BIOMASS": 5, # reward for increasing biomass
#     "HEALTH": 1, # reward for defending health
#     "REPRODUCTION": 15, # reward for reproduction
# }


@jdc.pytree_dataclass
class AgentState:
    """
    Represents the state of a single agent.
    
    Species ID: 0 for Tree, 1 for Fungus.
    Position: (x, y) coordinates on the grid.
    Biomass: Amount of biomass the agent has.
    Energy: Energy available for actions.
    Phosphorus: Amount of phosphorus available for sugar production.
    Sugars: Amount of sugars available for energy production.
    Health: Health of the agent.
    Defence: Defence level of the agent, affects health loss from herbivory and 
             pathogens.
    """
    species_id: jax.Array # 0 for Tree, 1 for fungus
    position: jax.Array  # (x, y) coordinates
    biomass: jax.Array = jdc.field(default_factory=lambda: jnp.array(0.1))
    phosphorus: jax.Array = jdc.field(default_factory=lambda: jnp.array(0.))
    sugars: jax.Array = jdc.field(default_factory=lambda: jnp.array(10.))
    health: jax.Array = jdc.field(default_factory=lambda: jnp.array(100.))
    defence: jax.Array = jdc.field(default_factory=lambda: jnp.array(1.0))
    radius: jax.Array = jdc.field(default_factory=lambda: jnp.array(0.0))


@jdc.pytree_dataclass
class EnvState:
    """
    Represents the full environment state with two AgentState instances for
    the Tree and Fungus.
    """
    grid: jax.Array   # (GRID_SIZE, GRID_SIZE) where values could be:
                      # 0 (empty), 1 (tree), 2 (fungus), 3 (fungus and tree)
    tree_agent: AgentState
    fungus_agent: AgentState
    step_count: jax.Array # To track episode length
    terminal: bool = False  # Flag to indicate if the episode is done
    # Solar irradiance for sugar production, can be modified for different scenarios
    solar_irradiance: jax.Array = jdc.field(default_factory=lambda: jnp.array(400.0))


class TwoSTwoR:
    """
    A minimal grid world environment with a single Tree and a single Fungus agent.
    Both have continuous resource allocation actions, with discrete numbers of sugars 
    required to produce seeds/fruiting bodies.
    
    Assume phosphorus is homogenously distributed in the environment, and both agents can
    access it but with different efficiencies.
    """
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.max_episode_steps = MAX_EPISODE_STEPS

    def _get_obs(self, state: EnvState):
        """
        Generates observations for the agents.
        Both agents observe their own state and overlapping positions with the other agent.
        """
        intersect = area_of_intersecting_circles(
            state.tree_agent.position, state.tree_agent.radius,
            state.fungus_agent.position, state.fungus_agent.radius
        ) > 0

        tree_obs = jnp.concatenate([
            state.tree_agent.position,
            state.tree_agent.phosphorus[None],  # Convert to 1D array
            state.tree_agent.sugars[None],
            state.tree_agent.health[None],
            jnp.array([intersect], dtype=jnp.int32) # Detect contact with fungus (1 if true, else 0)
        ])
        fungus_obs = jnp.concatenate([
            state.fungus_agent.position,
            state.fungus_agent.phosphorus[None],
            state.fungus_agent.sugars[None],
            state.fungus_agent.health[None],
            jnp.array([intersect], dtype=jnp.int32) # Detect contact with tree (1 if true, else 0)
        ])
        return {'tree': tree_obs, 'fungus': fungus_obs}

    def reset(self, key: jax.Array):
        """
        Resets the environment to an initial state, placing the Tree and Fungus at or 
        near the grid centre.
        """
        key, fungus_key = jax.random.split(key)

        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        grid_centre=jnp.array([self.grid_size // 2, self.grid_size // 2])

        fungus_pos_offset = jax.random.randint(fungus_key, (2,), 0, 1)

        tree_agent_state = AgentState(
            species_id=jnp.array(0),
            position=grid_centre,  # tree starts in the center
        )
        fungus_agent_state = AgentState(
            species_id=jnp.array(1),
            position=jnp.array(    # fungus starts at a random position near the center
                [grid_centre[0] + fungus_pos_offset[0],
                 grid_centre[1] + fungus_pos_offset[1]]),
        )

        # Mark positions on grid – tree = 1, fungus = 2, both = 3 if they overlap
        tree_pos = tuple(tree_agent_state.position)
        fungus_pos = tuple(fungus_agent_state.position)

        grid = grid.at[tree_pos].set(1)
        grid = grid.at[fungus_pos].set(
            jnp.where(jnp.all(tree_agent_state.position == fungus_agent_state.position), 3, 2)
        )

        state = EnvState(
            grid=grid,
            tree_agent=tree_agent_state,
            fungus_agent=fungus_agent_state,
            step_count=jnp.array(0, dtype=jnp.int32)
        )

        obs = self._get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def step_env(self, key: jax.Array, state: EnvState, actions: Dict[str, Dict[str, jax.Array]]):
        """
        Applies agent actions to the environment and updates the state.
        
        actions variable has the form: 
        {'tree': {'p_use': float, ' p_trade': float, 's_use': float, 's_trade': float,
                  'growth': float, 'defence': float, 'reproduction': float},
         'fungus': {'p_use': float, ' p_trade': float, 's_use': float, 's_trade': float,
                  'growth': float, 'defence': float, 'reproduction': float}}

        where each action is a continuous value representing the amount of energy allocated.
        """
        # Constrain allocations within available resources for both agents.
        actions['tree'] = self.constrain_allocation(actions['tree'])
        actions['fungus'] = self.constrain_allocation(actions['fungus'])

        # Allocate resources based on agent actions, convert from proportions to absolute values.
        actions['tree'] = self.allocate_resources(state.tree_agent, actions['tree'])
        actions['fungus'] = self.allocate_resources(state.fungus_agent, actions['fungus'])

        # Step each agent type – is it a problem that it's in sequence?
        state, tree_reward, tree_shaped_reward = self.step_tree(key, state, actions['tree'])
        state, fungus_reward, fungus_shaped_reward = self.step_fungus(key, state, actions['fungus'])
        state = self.step_trade(state, actions)

        # Update grid for new overlapping positions.
        # Not implemented...

        state = jdc.replace(state, step_count=state.step_count + 1)

        done = self.is_terminal(state)
        state = jdc.replace(state, terminal=done)

        obs = self._get_obs(state)
        rewards = {"tree": tree_reward, "fungus": fungus_reward}
        shaped_reward = {"tree": tree_shaped_reward, "fungus": fungus_shaped_reward}
        dones = {"tree": done, "fungus": done, "__all__": done}

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {'shaped_reward': shaped_reward},
        )

    def step_tree(self, key: jax.Array, state: EnvState, actions: Dict[str, jax.Array]):
        """Step the Tree agent based on its actions."""

        def biomass_to_canopy_area_allometry(biomass: jax.Array):
            """
            Convert biomass to canopy area using allometric scaling for European
            Beech tree. 
            From "Allometric Relationships of Selected European Tree Species" by 
            Widlowski et al. (2003), pg. 27, EUROPEAN COMMISSION JOINT RESEARCH CENTRE.
            """
            dbh = biomass ** (1/2.601) / 0.0798
            C_r = 0.0821 * dbh + 0.7694
            A_c = jnp.pi * C_r**2  # Area in square meters
            return A_c, C_r

        def gen_sugars(p: jax.Array, A_c: jax.Array, I_s: jax.Array):
            """
            Generate sugars based on the amount of phosphorus used and sunlight available.
            Uses Michaelis-Menten-like function for more realistic behaviour at irradiation
            saturation.
            p: Amount of phosphorus units used for sugar production.
            A_c: Canopy area of the tree.
            Returns: Amount of sugars generated.
            """
            S_max = 1200. # Maximum sugar production rate
            K_I = 400. # Half-saturation constant for sugar production
            s_gen = jnp.floor(A_c * S_max * (I_s / (K_I + I_s)))
            s_gen_clipped = jnp.clip(s_gen, 0, p / 3) # Hard limit: 3 P required per sugar
            return s_gen_clipped

        # --- Calculate resource usage ---
        # Calculate number of seeds that can be produced based on reproduction energy
        # and effective reproduction cost (discrete number of seeds).
        seeds_generated = actions['reproduction'] // SEED_COST
        effective_reproduction_cost = seeds_generated * SEED_COST

        # Recalculate amount of sugars used with effective reproduction cost
        s_used = actions['growth'] + actions['defence'] + effective_reproduction_cost

        # ---  Resource absorption  ---
        # Phosphorus absorption based on root area (same as canopy area for simplicity)
        A_c, r_c = biomass_to_canopy_area_allometry(state.tree_agent.biomass)
        p_acquired = jnp.floor(P_AVAILABILITY * TREE_P_UPTAKE_EFFICIENCY * A_c)

        # Sugars generated from sunlight, constrained by available phosphorus
        s_gen = gen_sugars(actions['p_use'], A_c, state.solar_irradiance)

        # --- Update tree agent state ---
        # Update health and defence
        defence_increase = actions['defence'] * DEFENCE_CONSTANT

        # Health loss proportional to growth (herbivory) and defence (pathogens).
        # Defence used currently from previous time step, not updated one.
        health_loss = (
            (PATHOGEN_ATTACK * actions['growth']) \
            / (state.tree_agent.health * state.tree_agent.defence)
        )

        # Update biomass based on growth
        biomass_increase = actions['growth'] * SUGARS_TO_BIOMASS

        # Update state with new values
        state = jdc.replace(state,
            tree_agent=jdc.replace(state.tree_agent,
                phosphorus=state.tree_agent.phosphorus + p_acquired - actions['p_use'],
                sugars=state.tree_agent.sugars + s_gen - s_used,
                health=state.tree_agent.health - health_loss,
                biomass=state.tree_agent.biomass + biomass_increase,
                defence=state.tree_agent.defence + defence_increase,
                radius=r_c,
            )
        )

        # Rewards for tree based on allocation (simplified)
        reward = 0.0
        shaped_rewards = {}

        reward += biomass_increase * 0.5 # Reward for growth
        reward += state.tree_agent.defence * 0.1 # Might not be necessary?
        reward += seeds_generated * 1.5 # Reward for each seed produced

        return state, reward, shaped_rewards

    def step_fungus(self, key: jax.Array, state: EnvState, actions: Dict[str, jax.Array]):

        def biomass_to_area_allometry(biomass: jax.Array):
            """
            Convert biomass to hyphae extent with reference to allometric scaling for 
            European Beech tree. A constant scaling factor for higher area per biomass 
            ratio for fungal agent than tree counterpart.

            From "Allometric Relationships of Selected European Tree Species" by 
            Widlowski et al. (2003), pg. 27, EUROPEAN COMMISSION JOINT RESEARCH CENTRE.
            """
            scaling_factor = 1.5  # Fungal hyphae spread further than tree canopy
            dbh = biomass ** (1/2.601) / 0.0798
            C_r = 0.0821 * dbh + 0.7694
            A_c = jnp.pi * C_r**2  # Area in square meters
            return A_c * scaling_factor, C_r * jnp.sqrt(scaling_factor)  # Return area and radius

        # --- Calculate resource usage ---
        # Calculate fruiting bodies produced based on reproduction energy.
        fruits_generated = actions['reproduction'] // SEED_COST
        effective_reproduction_cost = fruits_generated * SEED_COST

        # Recalculate amount of sugars used with effective reproduction cost
        s_used = actions['growth'] + actions['defence'] + effective_reproduction_cost

        # ---  Resource absorption  ---
        # Phosphorus absorption based on root area (same as canopy area for simplicity)
        A_c, r_c = biomass_to_area_allometry(state.fungus_agent.biomass)
        p_acquired = jnp.floor(P_AVAILABILITY * FUNGUS_P_UPTAKE_EFFICIENCY * A_c)

        # --- Update tree agent state ---
        # Update health and defence
        defence_increase = actions['defence'] * DEFENCE_CONSTANT

        # Health loss proportional to growth (herbivory) and defence (pathogens).
        # Defence used currently from previous time step, not updated one.
        health_loss = (
            (PATHOGEN_ATTACK * state.fungus_agent.biomass) \
            / (state.fungus_agent.health * state.fungus_agent.defence)
        )

        # Update biomass based on growth
        biomass_increase = actions['growth'] * SUGARS_TO_BIOMASS

        # Update state with new values
        state = jdc.replace(state,
            fungus_agent=jdc.replace(state.fungus_agent,
                phosphorus=state.fungus_agent.phosphorus + p_acquired,
                sugars=state.fungus_agent.sugars - s_used,
                health=state.fungus_agent.health - health_loss,
                biomass=state.fungus_agent.biomass + biomass_increase,
                defence=state.fungus_agent.defence + defence_increase,
                radius=r_c
            )
        )

        # Rewards for fungus based on allocation.
        reward = 0.0
        shaped_rewards = {}

        reward += biomass_increase * 0.5 # Reward for growth
        reward += state.fungus_agent.defence * 0.1 # Might not be necessary?
        reward += fruits_generated * 1.5 # Reward for each fruiting body produced

        return state, reward, shaped_rewards

    def step_trade(self, state: EnvState, actions: Dict[str, Dict[str, jax.Array]]):
        """
        Handle trade actions between Tree and Fungus agents.
        
        Will this work algorthmically if there is no explicit reward given for this step?
        """
        tree_actions = actions['tree']
        fungus_actions = actions['fungus']

        # ---  Handle trade actions  ---
        # Clip trades for limited phosphorus and sugar supply.
        # contact_area = jnp.where(state.grid == 3, 1, 0).sum(dtype=jnp.float32) # Count overlapping cells
        contact_area = area_of_intersecting_circles(
            state.tree_agent.position, state.tree_agent.radius,
            state.fungus_agent.position, state.fungus_agent.radius
        )

        def cap_trade(x):
            """Set maximum trade based on contact area."""
            max_trade = jnp.floor(contact_area * TRADE_PER_UNIT_AREA)
            return jnp.clip(x, 0, max_trade)

        # Cap trades to maximum allowed per contact cell
        # Tree -> fungus trade
        tf_p_trade = cap_trade(tree_actions['p_trade'])
        tf_s_trade = cap_trade(tree_actions['s_trade'])

        # Fungus -> tree trade
        ft_p_trade = cap_trade(fungus_actions['p_trade'])
        ft_s_trade = cap_trade(fungus_actions['s_trade'])

        state = jdc.replace(state,
            tree_agent=jdc.replace(state.tree_agent,
            phosphorus=state.tree_agent.phosphorus - tf_p_trade + ft_p_trade,
            sugars=state.tree_agent.sugars - tf_s_trade + ft_s_trade,
        ))
        state = jdc.replace(state,
            fungus_agent=jdc.replace(state.fungus_agent,
            phosphorus=state.fungus_agent.phosphorus - ft_p_trade + tf_p_trade,
            sugars=state.fungus_agent.sugars - ft_s_trade + tf_s_trade,
        ))
        return state

    def constrain_allocation(self, actions: Dict[str, jax.Array]):
        """
        Constrain resource allocation to ensure they sum to 1 or less.
        This is used for both Tree and Fungus agents.
        """
        def constrain_use_trade(u, t):
            """Constrain use and trade actions to be within available resources."""
            total = u + t
            return u / total, t / total

        def constrain_resource_allocation(g, d, r):
            """Constrain resource allocation to ensure they sum to 1 or less."""
            total = g + d + r
            return g / total, d / total, r / total

        # Prevent negative values.
        p_use = jnp.clip(actions['p_use'], 0)
        p_trade = jnp.clip(actions['p_trade'], 0)
        s_use = jnp.clip(actions['s_use'], 0)
        s_trade = jnp.clip(actions['s_trade'], 0)
        growth = jnp.clip(actions['growth'], 0)
        defence = jnp.clip(actions['defence'], 0)
        reproduction = jnp.clip(actions['reproduction'], 0)

        # Constrain use and trade actions for sugars and phosphorus.
        p_use, p_trade = jax.lax.cond(
            p_use + p_trade > 1,
            lambda x: constrain_use_trade(*x),
            lambda x: x,
            (p_use, p_trade)
        )
        s_use, s_trade = jax.lax.cond(
            s_use + s_trade > 1,
            lambda x: constrain_use_trade(*x),
            lambda x: x,
            (s_use, s_trade)
        )

        # Constrain resource allocation to ensure they sum to 1 or less.
        growth, defence, reproduction = jax.lax.cond(
            growth + defence + reproduction > 1,
            lambda x: constrain_resource_allocation(*x),
            lambda x: x,
            (growth, defence, reproduction)
        )

        # Update actions with constrained values
        actions['p_use'] = actions['p_use'].at[()].set(p_use) # () for scalar
        actions['p_trade'] = actions['p_trade'].at[()].set(p_trade)
        actions['s_use'] = actions['s_use'].at[()].set(s_use)
        actions['s_trade'] = actions['s_trade'].at[()].set(s_trade)
        actions['growth'] = actions['growth'].at[()].set(growth)
        actions['defence'] = actions['defence'].at[()].set(defence)
        actions['reproduction'] = actions['reproduction'].at[()].set(reproduction)

        return actions

    def allocate_resources(
            self, agent: AgentState, agent_actions: Dict[str, jax.Array],
        ) -> Dict[str, jax.Array]:
        """Allocate resources based on agent budget (ratios) allocation actions."""

        p_use = jnp.floor(agent_actions['p_use'] * agent.phosphorus)
        s_use = agent_actions['s_use'] * agent.sugars
        p_trade = jnp.floor(agent_actions['p_trade'] * agent.phosphorus)
        s_trade = jnp.floor(agent_actions['s_trade'] * agent.sugars)

        growth = jnp.floor(agent_actions['growth'] * s_use)
        defence = jnp.floor(agent_actions['defence'] * s_use)
        reproduction = jnp.floor(agent_actions['reproduction'] * s_use)

        agent_actions['p_use'] = agent_actions['p_use'].at[()].set(p_use) # () for scalar
        agent_actions['p_trade'] = agent_actions['p_trade'].at[()].set(p_trade)
        agent_actions['s_use'] = agent_actions['s_use'].at[()].set(s_use)
        agent_actions['s_trade'] = agent_actions['s_trade'].at[()].set(s_trade)
        agent_actions['growth'] = agent_actions['growth'].at[()].set(growth)
        agent_actions['defence'] = agent_actions['defence'].at[()].set(defence)
        agent_actions['reproduction'] = agent_actions['reproduction'].at[()].set(reproduction)

        return agent_actions

    def is_terminal(self, state: EnvState):
        """Check if the episode is done based on the step count or tree health."""
        return state.terminal | \
               (state.step_count >= self.max_episode_steps) | \
               (state.tree_agent.health <= 0) | \
               (state.fungus_agent.health <= 0) | \
               (jnp.where(state.grid == 1, 1, 0).sum() == 25) | \
               (jnp.where(state.grid == 2, 1, 0).sum() == 25) | \
               (jnp.where(state.grid == 3, 1, 0).sum() == 25)


if __name__ == "__main__":
    pass
