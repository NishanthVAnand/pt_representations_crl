import jax
import jax.numpy as jnp
import chex
import numpy as np

from functools import partial

@chex.dataclass(frozen=True)
class EnvState:
    state: jnp.ndarray  # Current state ID (int)
    timestep: jnp.ndarray  # Current timestep (int)
    rewards_history: jnp.ndarray # rewards history

@chex.dataclass(frozen=True)
class Observation:
    image: jnp.ndarray  # (32, 32, 3) image array
    tokens: jnp.ndarray  # Hash value 
    state_id: jnp.ndarray  # The state ID (raw state)

@chex.dataclass(frozen=True)
class EnvParams:
    num_samples: int # number of samples per class
    N: int = 10       # Number of states and actions
    p: float = 0.8    # Probability of staying in the same state
    T: int = 100      # Timesteps per non-stationarity phase
    num_phases: int = 3 # Number of non-stationarity phases
    noise_std: float = 0.01 # Std dev of reward noise
    slippery_p: float = 0.1 # Probability of action slip
    window: int = 10 # window size for moving average

@chex.dataclass(frozen=True)
class EnvData:
    images_all: jnp.ndarray # storing all images
    tokens_all: jnp.ndarray # storing all tokens

def make_env(data_path, N=10, p=0.8, T=100, num_phases=3, noise_std=0.01,
              slippery_p=0.1, window=10, num_features=128, num_images=500):
    
    try:
        data = np.load(data_path+"_"+str(num_features)+"_"+str(num_images)+".npz")
        np_images = data['images']
        np_tokens = data['tokens']
        print("Data loaded successfully.")
    
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return None, None, None

    size_per_class = np_images.shape[1]
    all_images = jnp.asarray(np_images)
    all_tokens = jnp.asarray(np_tokens)

    env_params = EnvParams(N=N, p=p, T=T, slippery_p=slippery_p, num_phases=num_phases, noise_std=noise_std,
                            num_samples=size_per_class, window=window)

    env_data = EnvData(images_all=all_images, tokens_all=all_tokens)

    def _sample_obs_from_state(rng: jax.random.PRNGKey, state: jnp.ndarray, env_params: EnvParams, env_data: EnvData) -> Observation:
        
        sample_state = jax.lax.cond(env_params.N <= 5, lambda state: state*2, lambda state: state, state)
        sample_idx = jax.random.randint(rng, shape=(), minval=0, maxval=env_params.num_samples)

        image = env_data.images_all[sample_state, sample_idx]
        token = env_data.tokens_all[sample_state, sample_idx]
        
        return Observation(image=image, tokens=token, state_id=state)

    def reset_fn(rng: jax.random.PRNGKey, env_params: EnvParams, env_data: EnvData) -> tuple[EnvState, Observation]:

        rng_state, rng_obs = jax.random.split(rng)
        
        start_state = jax.random.randint(rng_state, shape=(), minval=0, maxval=env_params.N)
        start_timestep = jnp.array(0)
        
        env_state = EnvState(state=start_state, timestep=start_timestep, rewards_history=jnp.full(env_params.window, jnp.nan))
        
        obs = _sample_obs_from_state(rng_obs, start_state, env_params, env_data)
        
        return env_state, obs

    def step_fn(rng: jax.random.PRNGKey, env_state: EnvState, action: jnp.ndarray, env_params: EnvParams, env_data: EnvData) -> tuple[EnvState, Observation, jnp.ndarray, jnp.ndarray, dict]:
        
        rng_act, rng_trans, rng_reward, rng_obs = jax.random.split(rng, 4)

        rng_act, rng_choice = jax.random.split(rng_act)
        rand_act_p = jax.random.uniform(rng_act)
        action = jnp.where(rand_act_p < 1 - env_params.slippery_p, action, jax.random.randint(rng_choice, shape=(), minval=0, maxval=env_params.N))
    
        shift = (env_state.timestep // env_params.T) % env_params.num_phases
        rewarding_action = (env_state.state + shift) % env_params.N
        is_rewarding = (action == rewarding_action)
        base_reward = jnp.where(is_rewarding, 1.0, -0.3)
        noise = jax.random.normal(rng_reward) * env_params.noise_std
        reward = base_reward + noise
        
        intended_state = jnp.where(is_rewarding, (env_state.state + 1) % env_params.N, env_state.state)
        failure_state = jnp.where(is_rewarding, env_state.state, (env_state.state + 1) % env_params.N)

        rand_trans = jax.random.uniform(rng_trans)
        next_state = jnp.where(rand_trans < env_params.p, intended_state, failure_state)

        new_env_state = env_state.replace(state=next_state, timestep=env_state.timestep + 1,
                                           rewards_history=jnp.roll(env_state.rewards_history, shift=-1, axis=-1).at[..., -1].set(reward))
        
        obs = _sample_obs_from_state(rng_obs, next_state, env_params, env_data)
        
        done = jnp.array(False)
        info = {"moving_avg_reward": jnp.nanmean(new_env_state.rewards_history)}
        
        return new_env_state, obs, reward, done, info
    
    return env_params, env_data, reset_fn, step_fn