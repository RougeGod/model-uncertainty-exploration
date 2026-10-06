from functools import partial

from gymnax import environments
from gymnax.environments import spaces, environment
from gymnax.environments.classic_control import mountain_car
from gymnax.wrappers.purerl import GymnaxWrapper


import jax
import jax.numpy as n


class NoiseObsWrapper(GymnaxWrapper):

    def __init__(self, env, noise_reward: float = 0.0):
        super().__init__(env)
        self._noise_reward = float(noise_reward)

    def action_space(self, params) -> spaces.Dict:
        return spaces.Dict({"env_action": self._env.action_space(params), "noise": spaces.Discrete(2)})


    @partial(jax.jit, static_argnames=["self"])
    def step(self, rng_key: jax.Array, state: environment.EnvState, action: spaces.Dict, params):
        key_step, key_reset = jax.random.split(rng_key)
        next_obs, next_state, reward, terminated, info = self.step_env(key_step, state, action, params)
        truncated = next_state.time >= params.max_steps_in_episode
        done = n.logical_or(terminated, truncated)
        reset_obs, reset_state = self._env.reset_env(key_reset, params)
    
        state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), reset_state, next_state)
        obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), reset_obs, next_obs)
        
        info = {**info, "terminated": terminated, "truncated": truncated, "final_observation": next_obs, "next_obs": next_obs}
        return obs, state, reward, terminated, truncated, info


    @partial(jax.jit, static_argnames=["self"])
    def step_env(self, rng_key: jax.Array, state: environment.EnvState,
             action: spaces.Dict, params: environment.EnvParams | None = None):
        next_obs, next_state, reward, done, info = jax.lax.cond(action["noise"], 
                                                   self.funky_dynamics, #true_fn 
                                                   self._env.step_env, #false_fn
                                                   rng_key, state, action["env_action"], params) #arguments to function
        return next_obs, next_state, reward, done, info

    @partial(jax.jit, static_argnames=["self"])
    def funky_dynamics(self, rng_key: jax.Array, env_state, action, params):
        next_obs = self.random_obs(rng_key, params)
        next_state = env_state.replace(time=env_state.time + 1) #freeze the environmental state
        reward = self._noise_reward
        done = False
        info = {"discount": self.discount(next_state, params)}
        return next_obs, next_state, reward, done, info
        

    @partial(jax.jit, static_argnames=["self"])
    def random_obs(self, rng_key, params, clip_value=1e6):
        rng_key, subkey = jax.random.split(rng_key)
        obs_dim = self._env.observation_space(params).shape
        high = n.clip(self._env.observation_space(params).high, -clip_value, clip_value)
        low = n.clip(self._env.observation_space(params).low, -clip_value, clip_value)
        return jax.random.uniform(rng_key, obs_dim, minval=low, maxval=high)


class FlattenActionWrapper(GymnaxWrapper):
    """Flatten a Dict action space into a single Discrete action space."""

    def action_space(self, params) -> spaces.Discrete:
        dict_space = self._env.action_space(params)
        num_actions = 1
        for sub_space in dict_space.spaces.values():
            num_actions *= sub_space.n
        return spaces.Discrete(num_actions)

    def _unflatten_action(self, action, params):
        dict_space = self._env.action_space(params)
        keys = list(dict_space.spaces.keys())
        sizes = tuple(dict_space.spaces[key].n for key in keys)
        indices = n.unravel_index(action, sizes)
        return {key: index for key, index in zip(keys, indices)}

    @partial(jax.jit, static_argnames=["self"])
    def step(self, rng_key: jax.Array, state: environment.EnvState, action: jax.Array, params=None):
        dict_action = self._unflatten_action(action, params)
        return self._env.step(rng_key, state, dict_action, params)

def make_noisy_mountain_car(noise_reward: float=-1.0, **env_kwargs):
    env = NoiseObsWrapper(mountain_car.MountainCar(**env_kwargs), noise_reward=noise_reward)
    return FlattenActionWrapper(env)

