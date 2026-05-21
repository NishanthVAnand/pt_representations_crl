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
import flax.linen as nn
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
from safetensors.flax import load_file, save_file

import wandb
import matplotlib.pyplot as plt

@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val_t: chex.Array
    q_val_p: chex.Array
    q_val: chex.Array

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
        x = nn.relu(x)
        x = nn.Dense(features=self.action_dim)(x)
        return x

class TNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=1024)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(features=1024)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.action_dim)(x)
        return x

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
    t_network = TNetwork(action_dim=env_params.N)

    lr = config.get("LR", 1e-3)
    t_lr = config.get("TRANSIENT_LR", 0.3)

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

            rng, _rng = jax.random.split(rng)
            t_network_vars = t_network.init(_rng, init_x)
            tx_transient = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=t_lr),
            )

            transient_train_state = CustomTrainState.create(
                apply_fn=t_network.apply,
                params=t_network_vars["params"],
                tx=tx_transient,
            )

            return train_state, transient_train_state

        rng, _rng = jax.random.split(rng)
        train_state, transient_train_state = create_agent(_rng)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, transient_train_state, env_state, obs, env_data, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, transient_train_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                last_img_obs = last_obs["image"].reshape(-1, 32 * 32 * 3).astype(jnp.float32) / 255.0

                q_vals_p = network.apply(
                    {
                        "params": train_state.params
                    },
                    last_img_obs
                )
                
                q_vals_t = t_network.apply(
                    {
                        "params": transient_train_state.params,
                    },
                    last_img_obs
                )

                q_vals = q_vals_p + q_vals_t  # combine the two Q-values

                # different eps for each env
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_env_state, new_obs, reward, new_done, info = step_fn(
                    rng_s, env_state, new_action, env_params, env_data
                )

                # update transient value function online
                new_img_obs = new_obs["image"].reshape(-1, 32 * 32 * 3).astype(jnp.float32) / 255.0
                q_vals_p_next = network.apply(
                    {
                        "params": train_state.params,
                    },
                    new_img_obs,
                )

                q_vals_t_next = t_network.apply(
                    {
                        "params": transient_train_state.params,
                    },
                    new_img_obs,
                )

                q_vals_next = q_vals_p_next + q_vals_t_next

                q_vals_p_action = jnp.take_along_axis(
                    q_vals_p,
                    jnp.expand_dims(new_action, axis=-1),
                    axis=-1,
                ).squeeze(axis=-1)

                new_done_f = new_done.astype(jnp.float32)  # convert to float for calculations
                targets = reward + config.get("GAMMA", 0.99) * (
                    1.0 - new_done_f
                ) * jnp.max(q_vals_next, axis=-1) - q_vals_p_action
                targets = jax.lax.stop_gradient(targets)

                def update_transient_step(transient_train_state, targets, inputs, actions):
                    def tloss_fn(params):
                        q_vals_t = t_network.apply(
                            {
                                "params": params,
                            },
                            inputs,
                        )
                        chosen_q_vals_t = jnp.take_along_axis(
                            q_vals_t,
                            jnp.expand_dims(actions, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)
                        
                        loss = 0.5 * jnp.square(chosen_q_vals_t - targets).mean()
                        return loss, None

                    (tloss, _), grads = jax.value_and_grad(
                        tloss_fn, has_aux=True
                    )(transient_train_state.params)
                    transient_train_state = transient_train_state.apply_gradients(
                        grads=grads
                    )
                    transient_train_state = transient_train_state.replace(
                        grad_steps=transient_train_state.grad_steps + 1,
                    )
                    return transient_train_state, tloss
                
                transient_train_state, _ = update_transient_step(
                    transient_train_state, targets, last_img_obs, new_action
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val_t=q_vals_t,
                    q_val_p=q_vals_p,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, transient_train_state, rng), (transition, info)

            # step the env
            rng, _rng = jax.random.split(rng)
            scan_carry = (obs, env_state, transient_train_state, _rng)
            (obs, env_state, transient_train_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                scan_carry,
                None,
                config["NUM_STEPS"],
            )

            train_state = train_state.replace(timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"])

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch):

                    train_state, rng = carry

                    def _loss_fn(params):

                        # permanent update
                        
                        all_q_vals_p = network.apply(
                            {
                                "params": params
                            },
                            minibatch.obs["image"].reshape(-1, 32 * 32 * 3).astype(jnp.float32) / 255.0
                        )

                        # Add extra dim to table and remove take_along_axis
                        all_q_vals_t = t_network.apply(
                            {
                                "params": transient_train_state.params
                            },
                            minibatch.obs["image"].reshape(-1, 32 * 32 * 3).astype(jnp.float32) / 255.0
                        )

                        all_q_vals_t_action = jnp.take_along_axis(
                            all_q_vals_t,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        old_q_vals_p_action = jnp.take_along_axis(
                            minibatch.q_val_p,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        target = old_q_vals_p_action + all_q_vals_t_action  # combine the two Q-values
                        target = jax.lax.stop_gradient(target)

                        chosen_action_qvals = jnp.take_along_axis(
                            all_q_vals_p,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, (None, chosen_action_qvals)

                    (loss, (_, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                    )
                    return (train_state, rng), (loss, qvals)

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
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), minibatches
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
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

            decay_factor = config.get("DECAY", 0.5)
            decay_fn = lambda x: x * decay_factor
            new_params = jax.tree_util.tree_map(decay_fn, transient_train_state.params)
            transient_train_state = transient_train_state.replace(params=new_params)
        
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

            runner_state = (train_state, transient_train_state, env_state, obs, env_data, rng)

            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        reset_keys = jax.random.split(_rng, config["NUM_ENVS"])
        vmap_reset = jax.vmap(reset_fn, in_axes=(0, None, None))
        env_state, obs = vmap_reset(reset_keys, env_params, env_data)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, transient_train_state, env_state, obs, env_data, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}
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