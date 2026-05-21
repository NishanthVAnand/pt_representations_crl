import os

import copy
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
from safetensors.flax import load_file, save_file

import wandb
import matplotlib.pyplot as plt

from DND import *

@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    done: chex.Array

class CustomTrainState(TrainState):
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0

class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=1024)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(features=1024)(x)
        x = nn.LayerNorm()(x)
        features = nn.relu(x)
        x = nn.Dense(features=self.action_dim)(features)
        return features, x

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

    network = QNetwork(action_dim=env_params.N)
    lr = config.get("LR", 1e-3)

    def train(rng):

        original_rng = rng[0]

        def create_agent(rng):
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros((1, 32 * 32 * 3), dtype=jnp.float32)
            network_vars = network.init(_rng, init_x)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_vars["params"],
                tx=tx
            )

            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(_rng)

        rng, _rng = jax.random.split(rng)
        dnd_buffer = init_DND(
                table_size = config["TRANSIENT_TABLE_SIZE"],
                hidden_size = 1024,
                num_actions = config["NUM_STATES"]
            )

        # TRAINING LOOP
        def _update_step(runner_state, _):

            train_state, dnd_buffer, env_state, obs, env_data, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, dnd_buffer, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                last_img_obs = last_obs["image"].reshape(-1, 32 * 32 * 3).astype(jnp.float32) / 255.0
                
                c_features, q_vals_net = network.apply(
                    {
                        "params": train_state.params
                    },
                    last_img_obs
                )

                batched_get = jax.vmap(get, in_axes=(None, 0))
                q_vals = batched_get(dnd_buffer, c_features)
                
                # different eps for each env
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                vmap_step = jax.vmap(step_fn, in_axes=(0, 0, 0, None, None))

                step_rngs = jax.random.split(rng_s, config["NUM_ENVS"])
                new_env_state, new_obs, reward, new_done, info = vmap_step(
                    step_rngs, env_state, new_action, env_params, env_data
                )

                new_img_obs = new_obs["image"].reshape(-1, 32 * 32 * 3).astype(jnp.float32) / 255.0
                n_features, _ = network.apply(
                    {
                        "params": train_state.params
                    },
                    new_img_obs
                )

                new_done_f = new_done.astype(jnp.float32)  
                
                detached_c_features = jax.lax.stop_gradient(c_features)
                detached_n_features = jax.lax.stop_gradient(n_features)
                
                def update_single_env(carry_table, env_idx):
                    updated_table = one_step_update(
                        carry_table, 
                        detached_c_features[env_idx], 
                        new_action[env_idx], 
                        reward[env_idx], 
                        detached_n_features[env_idx],
                        new_done_f[env_idx],
                        q_vals[env_idx], 
                        config["GAMMA"], 
                        config["NEC_LR"], 
                        config["NEC_SIM"]
                    )
                    return updated_table, None

                dnd_buffer, _ = jax.lax.scan(
                    update_single_env, 
                    dnd_buffer, 
                    jnp.arange(config["NUM_ENVS"])
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward = reward,
                    next_obs = new_obs,
                    done = new_done_f
                )

                return (new_obs, new_env_state, dnd_buffer, rng), (transition, info)

            # step the env
            rng, _rng = jax.random.split(rng)
            scan_carry = (obs, env_state, dnd_buffer, _rng)
            (obs, env_state, dnd_buffer, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                scan_carry,
                None,
                config["NUM_STEPS"],
            )

            train_state = dc.replace(train_state,
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, dnd_buffer, rng = carry

                def _learn_phase(carry, minibatch):

                    train_state, dnd_buffer, rng = carry

                    def _loss_fn(params):
                        
                        all_features, _ = network.apply(
                            {
                                "params": params
                            },
                            minibatch.obs["image"].reshape(-1, 32 * 32 * 3).astype(jnp.float32) / 255.0
                        )

                        batched_get = jax.vmap(get, in_axes=(None, 0))
                        # Read directly from the dnd_buffer in the closure
                        all_q_vals = batched_get(dnd_buffer, all_features)

                        all_q_vals_chosen = jnp.take_along_axis(
                            all_q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        next_features, _ = network.apply(
                            {"params": params},
                            minibatch.next_obs["image"].reshape(-1, 32 * 32 * 3).astype(jnp.float32) / 255.0
                        )
                        
                        next_features = jax.lax.stop_gradient(next_features)
                        next_q_vals = batched_get(dnd_buffer, next_features)
                        max_next_q = jnp.max(next_q_vals, axis=-1)
                        
                        target = minibatch.reward + config["GAMMA"] * max_next_q * (1.0 - minibatch.done)
                        target = jax.lax.stop_gradient(target)

                        loss = 0.5 * jnp.square(all_q_vals_chosen - target).mean()

                        return loss, (None, all_q_vals_chosen)

                    (loss, (_, qvals)), grads = jax.value_and_grad(
                        _loss_fn, argnums=0, has_aux=True
                    )(train_state.params)
                    
                    train_state = train_state.apply_gradients(grads=grads)

                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1
                    )
                    return (train_state, dnd_buffer, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = x.reshape(
                        -1, *x.shape[2:]
                    )  # num_steps*num_envs (batch_size), ...
                    x = jax.random.permutation(rng, x)  # shuffle the transitions
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...

                rng, _rng = jax.random.split(rng)
                (train_state, dnd_buffer, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, dnd_buffer, rng), minibatches
                )

                return (train_state, dnd_buffer, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, dnd_buffer, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, dnd_buffer, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "timesteps": train_state.timesteps,
                "avg_reward": infos["moving_avg_reward"][-1],
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
            }

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

            runner_state = (train_state, dnd_buffer, env_state, obs, env_data, rng)

            return runner_state, metrics


        rng, _rng = jax.random.split(rng)
        reset_keys = jax.random.split(_rng, config["NUM_ENVS"])
        vmap_reset = jax.vmap(reset_fn, in_axes=(0, None, None))
        env_state, obs = vmap_reset(reset_keys, env_params, env_data)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, dnd_buffer, env_state, obs, env_data, _rng)

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
        group=config["GROUP"],
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