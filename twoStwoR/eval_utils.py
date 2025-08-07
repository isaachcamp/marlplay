
import numpy as np
import jax.numpy as jnp
import jax

from twoStwoR import TwoSTwoR, ppo


def extract_episodes(qoi_traj: jnp.ndarray, done_traj: jnp.ndarray):
    """
    Extracts complete and incomplete episodes from batched trajectory data.

    Args:
        qoi_traj: JAX array of quantity of interest (QOI) with shape (NUM_UPDATES, NUM_STEPS, NUM_ENVS).
        done_traj: JAX array of done flags with shape (NUM_UPDATES, NUM_STEPS, NUM_ENVS).

    Returns:
        A list of lists, where each inner list contains numpy arrays of QOIs for
        episodes from a single environment.
    """
    num_updates, num_steps, num_envs = qoi_traj.shape
    total_steps = num_updates * num_steps

    # Flatten and convert to numpy for efficient, non-JIT analysis
    qoi_np = np.array(qoi_traj.reshape(total_steps, num_envs))
    done_np = np.array(done_traj.reshape(total_steps, num_envs))

    episode_qoi_per_env = [[] for _ in range(num_envs)]

    # Find the indices of done flags for each environment
    for env_idx in range(num_envs):
        done_indices_env = np.where(done_np[:, env_idx])[0]

        last_done_step = -1
        # Extract rewards for completed episodes
        for done_step_idx in done_indices_env:

            start_step = last_done_step + 1
            episode_qoi = qoi_np[start_step : done_step_idx + 1, env_idx]
            if episode_qoi.size > 0:
                episode_qoi_per_env[env_idx].append(episode_qoi)
            last_done_step = done_step_idx

        # Handle the final, incomplete episode
        if last_done_step + 1 < total_steps:
            remaining_qoi = qoi_np[last_done_step + 1:, env_idx]
            if remaining_qoi.size > 0:
                episode_qoi_per_env[env_idx].append(remaining_qoi)

    max_episode_number = max(len(episodes) for episodes in episode_qoi_per_env)
    max_episode_length = max(len(episode) for env_episodes in episode_qoi_per_env for episode in env_episodes)
    qoi_arr = np.zeros((num_envs, max_episode_number, max_episode_length))

    for env_idx, env_episodes in enumerate(episode_qoi_per_env):
        for episode_idx, episode in enumerate(env_episodes):
            qoi_arr[env_idx, episode_idx, :len(episode)] = episode
            if len(episode) < max_episode_length:
                qoi_arr[env_idx, episode_idx, len(episode):] = np.nan

    return qoi_arr

def collect_eval_traj(rng, config, train_state):
    max_episode_steps = 1000
    env = TwoSTwoR(grid_size=config["GRID_SIZE"], max_episode_steps=max_episode_steps)
    first_obs, env_state = env.reset(rng)

    def env_step(runner_state, x):
        train_state, env_state, last_obs, rng = runner_state
        rng, tree_act_rng, fungus_act_rng = jax.random.split(rng, 3)

        obs_batch = ppo.batchify(last_obs, env.agents, 1, config["NUM_ACTORS"])
        tree_pi, _ = train_state['tree'].apply_fn(train_state["tree"].params, obs_batch[0])
        fungus_pi, _ = train_state['fungus'].apply_fn(train_state["fungus"].params, obs_batch[1])
    
        tree_action = tree_pi.sample(seed=tree_act_rng)
        fungus_action = fungus_pi.sample(seed=fungus_act_rng)

        env_act = ppo.unbatchify(
            jnp.stack([tree_action, fungus_action]),
            env.agents, 1, config["NUM_ACTORS"]
        )
        env_act = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), env_act)

        rng, rng_step = jax.random.split(rng)
        obs, env_state, _, done, info = env.step_env(rng_step, env_state, env_act)

        runner_state = (train_state, env_state, obs, rng)
        return runner_state, (env_state, done, info['shaped_reward'])

    runner_state = (train_state, env_state, first_obs, rng)
    runner_state, (env_state, done, info) = jax.lax.scan(
        env_step, runner_state, None, max_episode_steps
    )
    return env_state, done, info

def last_complete_episode(episodes: jnp.ndarray) -> jnp.ndarray:
    """
    Returns last complete episode from 2D array of shape (NUM_EPISODES, EPISODE_LENGTH).
    If no complete episode exists, returns an empty array.
    """
    complete_episodes = episodes[~jnp.isnan(episodes).any(axis=-1)]
    if complete_episodes.size == 0:
        return jnp.array([])  # Return empty array if no complete episodes
    return complete_episodes[-1] # Return the last complete episode

def _normalise_actions(actions: np.ndarray) -> np.ndarray:
    """Constrains actions to be non-negative and normalized to sum to 1."""
    actions = actions / np.sum(actions, axis=0, keepdims=True)
    return actions

def constrain_actions(actions: np.ndarray) -> np.ndarray:
    """Conditionally normalises actions if the sum is greater than 1."""
    return np.where(
        np.sum(actions, axis=0) > 1.0,
        _normalise_actions(actions),
        actions
    )
