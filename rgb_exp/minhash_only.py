"""
Different hashing --- Minhash.
"""

import os

import copy
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import optax
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
from safetensors.flax import load_file, save_file

import wandb
import matplotlib.pyplot as plt

from transient_approx import *

@chex.dataclass
class CustomTrainState:
    timesteps: int = 0
    n_updates: int = 0

def make_train(config):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    if config["ENV_NAME"] == "circular-mdp":
        from circular_mdp import make_env
    elif config["ENV_NAME"] == "image-mdp-v1":
        from image_mdp_v1 import make_env
    elif config["ENV_NAME"] == "image-mdp-v2":
        from image_mdp_v2 import make_env
    else:
        raise NotImplementedError(f"Env {config['ENV_NAME']} not implemented.")

    # create environment
    env_params, env_data, reset_fn, step_fn = make_env(
        data_path=config["DATA_PATH"],
        N=config.get("NUM_STATES", 10),
        p=config.get("STAY_PROB", 0.8),
        T=config.get("PHASE_LENGTH", 100),
        num_phases=config.get("NUM_PHASES", 3),
        noise_std=config.get("REWARD_NOISE_STD", 0.01),
        slippery_p=config.get("SLIPPERY_PROB", 0.1),
        window=config.get("MOVING_AVG_WINDOW", 10),
        num_features=config.get("NUM_FEATURES", 256),
        num_images=config.get("NUM_IMGS", 250),
    )

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions
    
    eps = jnp.full((config["NUM_ENVS"],), config.get("EPSILON", 0.1))

    def train(rng):

        original_rng = rng[0]

        rng, _rng = jax.random.split(rng)
        
        transient_table = init_minhash_lsh_cache(
                rng=_rng,
                num_actions=config["NUM_STATES"],
                num_bins=config["TRANSIENT_TABLE_SIZE"],
                num_tables=config["NUM_HASHES"],
                rows_per_band=config["NUM_ROWS"],
                slots_per_bin=config["SLOTS_PER_BIN"],
                lambda_wt=config["RHO"],
            )
        
        train_state = CustomTrainState(
            timesteps=0,
            n_updates=0,
        )

        # TRAINING LOOP
        def _update_step(runner_state, _):

            train_state, transient_table, env_state, obs, env_data, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, table, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                
                table = put(table, last_obs["tokens"])
                q_vals, _ = get(table, last_obs["tokens"])

                # different eps for each env
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_env_state, new_obs, reward, new_done, info = step_fn(
                    rng_s, env_state, new_action, env_params, env_data
                )
                
                table = put(table, new_obs["tokens"])
                q_vals_next, _ = get(table, new_obs["tokens"])

                new_done_f = new_done.astype(jnp.float32)  # convert to float for calculations
                td_error = reward + (1 - new_done_f) * config["GAMMA"] * jnp.max(q_vals_next, axis=-1) - \
                    jnp.take_along_axis(q_vals, jnp.expand_dims(new_action, axis=-1), axis=-1).squeeze(axis=-1)
                td_error = jax.lax.stop_gradient(td_error)

                table = td_update_hit_only(table, last_obs["tokens"], new_action, td_error, config["TRANSIENT_LR"])

                return (new_obs, new_env_state, table, rng), info

            # step the env
            rng, _rng = jax.random.split(rng)
            scan_carry = (obs, env_state, transient_table, _rng)
            (obs, env_state, transient_table, rng), info = jax.lax.scan(
                _step_env,
                scan_carry,
                None,
                config["NUM_STEPS"],
            )

            train_state = dc.replace(train_state,
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"],
                n_updates=train_state.n_updates + 1,
            )

            metrics = {
                "update_steps": train_state.n_updates,
                "timesteps": train_state.timesteps,
                "avg_reward": info["moving_avg_reward"][-1],
            }

            transient_table = decay_values(transient_table, config["DECAY"])

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_rng, q_table, n_updates):
                    if (
                        n_updates % config.get("WANDB_LOG_INTERVAL", 128) == 0
                    ):
            
                        if config.get("WANDB_LOG_ALL_SEEDS", False):
                            metrics.update(
                                {
                                    f"rng{int(original_rng)}/{k}": v
                                    for k, v in metrics.items()
                                }
                            )
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_rng, None, train_state.n_updates)

            runner_state = (train_state, transient_table, env_state, obs, env_data, rng)

            return runner_state, metrics


        rng, _rng = jax.random.split(rng)
        reset_keys = jax.random.split(_rng, config["NUM_ENVS"])
        vmap_reset = jax.vmap(reset_fn, in_axes=(0, None, None))
        env_state, obs = vmap_reset(reset_keys, env_params, env_data)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, transient_table, env_state, obs, env_data, _rng)

        runner_state, metrics_history = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics_history}
    return train


def single_run(config):

    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        group=config.get("GROUP", None),
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config.get("NAME", f'{config["ALG_NAME"]}_{config["ENV_NAME"]}'),
        config=config,
        mode=config["WANDB_MODE"],
    )
    wandb.define_metric("timesteps")
    wandb.define_metric("*", step_metric="timesteps")

    wandb.define_metric("avg_reward", summary="mean")

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    # outs = jax.vmap(make_train(config))(rngs)
    print(f"Took {time.time()-t0} seconds to complete.")

    if config.get("SAVE_PATH", None) is not None:
        from purejaxql.utils.save_load import save_params
        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        # Inside single_run, in the SAVE_PATH block:
        for i, rng in enumerate(rngs):
            # The table is the 2nd element (index 1) of the runner_state tuple
            table_to_save = jax.tree_util.tree_map(lambda x: x[i], outs["runner_state"][1])
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_file(table_to_save, save_path)

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    single_run(config)

if __name__ == "__main__":
    main()