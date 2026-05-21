"""
This script uses BatchRenorm for more effective batch normalization in long training runs.
"""

import os
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"

import copy
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Optional

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax import serialization
import hydra
from omegaconf import OmegaConf
from safetensors.flax import load_file, save_file

import wandb

from craftax.craftax_env import make_craftax_env_from_name
from purejaxql.utils.craftax_wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
)
from purejaxql.utils.batch_renorm import BatchRenorm


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    norm_type: str = "batch_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = BatchRenorm(use_running_average=not train)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)

        return x


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array

class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
    old_params: Any = None
    fisher: Any = None

def make_train(config, initial_runner_state: Optional[Any] = None):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    basic_env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = basic_env.default_params
    log_env = LogWrapper(basic_env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
        test_env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["TEST_NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["TEST_NUM_ENVS"]),
        )
    else:
        env = BatchEnvWrapper(log_env, num_envs=config["NUM_ENVS"])
        test_env = BatchEnvWrapper(log_env, num_envs=config["TEST_NUM_ENVS"])

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
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
    
    network = QNetwork(
        action_dim=env.action_space(env_params).n,
        hidden_size=config.get("HIDDEN_SIZE", 128),
        num_layers=config.get("NUM_LAYERS", 2),
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
    )

    lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
    lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]    

    def compute_empirical_fisher(params, batch_stats, minibatch):
    
        def single_sample_loss(params, obs, next_obs, action, reward, done):
            variables = {"params": params, "batch_stats": batch_stats}
            q_vals = network.apply(variables, obs, train=False)
            q_vals_next = network.apply(variables, next_obs, train=False)
            
            chosen_q = q_vals[action]
            max_next_q = jnp.max(q_vals_next)
            
            target = reward + config["GAMMA"] * (1.0 - done) * max_next_q
            target = jax.lax.stop_gradient(target)
            
            return 0.5 * jnp.square(chosen_q - target)

        grad_fn = jax.vmap(jax.grad(single_sample_loss), in_axes=(None, 0, 0, 0, 0, 0))
        
        batch_grads = grad_fn(
            params, 
            minibatch.obs, 
            minibatch.next_obs, 
            minibatch.action, 
            minibatch.reward, 
            minibatch.done
        )
        
        fisher = jax.tree_util.tree_map(lambda g: jnp.mean(jnp.square(g), axis=0), batch_grads)
        return fisher

    def create_agent(rng):
    
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_variables = network.init(_rng, init_x, train=False)
        params = network_variables["params"]
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )

        zero_pytree = jax.tree_util.tree_map(jnp.zeros_like, params)

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_variables["params"],
            batch_stats=network_variables["batch_stats"],
            tx=tx,
            old_params=params, # First task anchors to its initialization
            fisher=zero_pytree
        )

        return train_state

    def _create_initial_runner_state(rng):
        print(f"[Seed {rng[0]}] Creating new runner state")
        rng, _rng = jax.random.split(rng)
        train_state = create_agent(_rng)

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        expl_state = env.reset(_rng, env_params)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, expl_state, test_metrics, _rng)
        return runner_state
    
    def get_test_metrics(train_state, rng):
            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(_rng, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = test_env.step(
                    _rng, env_state, new_action, env_params
                )
                return (new_env_state, new_obs, rng), info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.reset(_rng, env_params)

            _, infos = jax.lax.scan(
                _env_step, (env_state, init_obs, _rng), None, config["TEST_NUM_STEPS"]
            )
            # return mean of done infos
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / infos["returned_episode"].sum(),
                infos,
            )
            return done_infos

    def train(rng, initial_runner_state_for_this_seed):

        original_rng = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )

                # different eps for each env
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, rng), (transition, info)

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            last_q = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                transitions.next_obs[-1],
                train=False,
            )
            last_q = jnp.max(last_q, axis=-1)

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (
                    1 - transition.done
                ) * lambda_returns + transition.done * transition.reward
                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            last_q = last_q * (1 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):

                        if config.get("Q_LAMBDA", False):
                            q_vals, updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                minibatch.obs,
                                train=True,
                                mutable=["batch_stats"],
                            )
                        else:
                            # if not using q_lambda, re-pass the next_obs through the network to compute target
                            all_q_vals, updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                jnp.concatenate((minibatch.obs, minibatch.next_obs)),
                                train=True,
                                mutable=["batch_stats"],
                            )
                            q_vals, q_next = jnp.split(all_q_vals, 2)
                            q_next = jax.lax.stop_gradient(q_next)
                            q_next = jnp.max(q_next, axis=-1)  # (batch_size,)
                            target = (
                                minibatch.reward
                                + (1 - minibatch.done) * config["GAMMA"] * q_next
                            )

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        td_loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        def calc_ewc_penalty(p, p_old, f):
                            return jnp.sum(f * jnp.square(p - p_old))
                        
                        ewc_penalties = jax.tree_util.tree_map(
                            calc_ewc_penalty, 
                            params, 
                            train_state.old_params, 
                            train_state.fisher
                        )
                        ewc_loss = sum(jax.tree_util.tree_leaves(ewc_penalties))

                        lambda_ewc = config.get("EWC_LAMBDA", 1000.0)
                        total_loss = td_loss + (lambda_ewc / 2.0) * ewc_loss

                        return total_loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(grad_steps=train_state.grad_steps + 1, batch_stats=updates["batch_stats"])
                    minibatch_fisher = compute_empirical_fisher(train_state.params, train_state.batch_stats, minibatch)
                    
                    return (train_state, rng), (loss, qvals, minibatch_fisher)

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
                targets = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals, stacked_fishers) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )

                mean_fisher = jax.tree_util.tree_map(
                    lambda x: jnp.mean(x, axis=0), stacked_fishers
                )
                
                fisher_alpha = config.get("FISHER_EMA_ALPHA", 0.99)
                anchor_alpha = config.get("ANCHOR_EMA_ALPHA", 0.999)

                new_fisher = jax.tree_util.tree_map(
                    lambda f_old, f_new: fisher_alpha * f_old + (1.0 - fisher_alpha) * f_new,
                    train_state.fisher, mean_fisher
                )

                new_old_params = jax.tree_util.tree_map(
                    lambda p_old, p_curr: anchor_alpha * p_old + (1.0 - anchor_alpha) * p_curr,
                    train_state.old_params, train_state.params
                )

                train_state = train_state.replace(fisher=new_fisher, old_params=new_old_params)

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / infos["returned_episode"].sum(),
                infos,
            )
            metrics.update(done_infos)

            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

            # remove achievement metrics if not logging them
            if not config.get("LOG_ACHIEVEMENTS", False):
                metrics = {
                    k: v for k, v in metrics.items() if "achievement" not in k.lower()
                }

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_rng):
                    
                    # log at intervals 
                    if (
                        metrics["update_steps"] % config.get("WANDB_LOG_INTERVAL", 128) == 0
                    ):
                        original_rng_int = original_rng.item()
                        if config.get("WANDB_LOG_ALL_SEEDS", False):
                            metrics.update(
                                {
                                    f"rng{int(original_rng_int)}/{k}": v
                                    for k, v in metrics.items()
                                }
                            )
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_rng)

            # CHECKPOINT SAVING
            checkpoint_interval = config.get("CHECKPOINT_INTERVAL", 0) # 0 to disable
            current_runner_state_to_save = (train_state, tuple(expl_state), test_metrics, rng)

            def _checkpoint_callback(runner_state_to_save, update_step, seed_rng):
                if checkpoint_interval > 0 and update_step % checkpoint_interval == 0 and update_step > 0:
                    seed_rng_val = seed_rng.item()

                    base_dir = os.path.join(wandb.run.dir, "checkpoints")
                    checkpoint_dir = os.path.join(base_dir, config["ALG_NAME"], f"seed_{seed_rng_val}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    print(f"\n[Seed {seed_rng_val}] Checkpointing at update step {update_step}")
                    
                    # Serialize the *entire* runner_state PyTree
                    bytes_data = serialization.to_bytes(runner_state_to_save)
                    
                    # Save 'latest' for easy resume
                    checkpoint_path_latest = os.path.join(checkpoint_dir, "checkpoint_latest.msgpack")
                    with open(checkpoint_path_latest, "wb") as f:
                        f.write(bytes_data)
                    
                    print(f"[Seed {seed_rng_val}] Checkpoint saved locally to {checkpoint_dir}")

            jax.debug.callback(_checkpoint_callback, 
                               current_runner_state_to_save, 
                               train_state.n_updates,
                               original_rng
                               )
            
            runner_state = (train_state, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        if initial_runner_state_for_this_seed:
            print("Using pre-loaded runner state")
            runner_state = initial_runner_state_for_this_seed

        else:
            print("Creating new runner state")
            runner_state = _create_initial_runner_state(rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return {"train_fn": train, "init_fn": _create_initial_runner_state}


def single_run(config):

    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]

    run_id = config.get("WANDB_RUN_ID", None)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config.get("NAME", f'{config["ALG_NAME"]}_{config["ENV_NAME"]}'),
        config=config,
        mode=config["WANDB_MODE"],
        id=run_id, # Set the run ID, if provided
        resume="allow", # Allow resuming
    )

    wandb.define_metric("env_step")
    wandb.define_metric("*", step_metric="env_step")

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    
    resume_dir = config.get("RESUME_FROM_CHECKPOINT_DIR", None)
    batched_initial_state = None
    
    if resume_dir:
        loaded_states = []
        all_loaded_successfully = True
        template_state = None # To hold the PyTree structure
        
        for i in range(config["NUM_SEEDS"]):
            seed_rng_val = rngs[i][0].item()
            
            checkpoint_file = os.path.join(resume_dir, config["ALG_NAME"], f"seed_{seed_rng_val}", "checkpoint_latest.msgpack")
            
            if not os.path.exists(checkpoint_file):
                print(f"[Seed {seed_rng_val}] Checkpoint not found")
                all_loaded_successfully = False
                break
            
            print(f"[Seed {seed_rng_val}] Loading from: {checkpoint_file}")
            try:
                if template_state is None:
                    train_fns = make_train(config, None)
                    template_state = jax.jit(train_fns["init_fn"])(rngs[0])
                
                with open(checkpoint_file, "rb") as f:
                    bytes_data = f.read()
                loaded_state = serialization.from_bytes(template_state, bytes_data)
                loaded_states.append(loaded_state)
                print(f"[Seed {seed_rng_val}] Resumed at update step {loaded_state[0].n_updates}")

            except Exception as e:
                print(f"[Seed {seed_rng_val}] ERROR loading: {e}")
                all_loaded_successfully = False
                break
        
        if all_loaded_successfully:
            print("All seeds loaded successfully. Stacking states")
            batched_initial_state = jax.tree.map(lambda *x: jnp.stack(x), *loaded_states)
        else:
            print("One or more seeds failed to load. Starting all seeds from scratch")
    else:
        print("No resume directory. Starting all seeds from scratch")

    train_vjit = jax.jit(jax.vmap(make_train(config, batched_initial_state)["train_fn"]))
    t0 = time.time()
    outs = jax.block_until_ready(train_vjit(rngs, batched_initial_state))

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

        for i, rng in enumerate(rngs):
            seed_rng_val = rngs[i][0].item()
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{seed_rng_val}_vmap{i}.safetensors',
            )
            save_params(params, save_path)

def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        wandb.define_metric("env_step")
        wandb.define_metric("*", step_metric="env_step")

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_fn = make_train(config, None)["train_fn"]
        train_vjit = jax.jit(jax.vmap(train_fn))
        outs = jax.block_until_ready(train_vjit(rngs, None))
        # return outs

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.0001
                ]
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
