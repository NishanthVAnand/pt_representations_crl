import jax
import jax.numpy as jnp
import chex
import dataclasses as dc

@chex.dataclass
class DND:
    values: jnp.ndarray   # (TABLES_SIZE, NUM_ACTIONS) float32
    keys: jnp.ndarray # (TABLES_SIZE, HIDDEN_SIZE) float32
    age:  jnp.ndarray   # (TABLES, 1) int32
    clock: jnp.ndarray   # () int32

def init_DND(table_size: int = 128, hidden_size: int = 128, num_actions = 4) -> DND:
    return DND(
        values = jnp.zeros((table_size, num_actions), dtype=jnp.float32),
        keys = jnp.zeros((table_size, hidden_size), dtype=jnp.float32),
        age = jnp.zeros((table_size), dtype=jnp.int32),
        clock = jnp.array(1, dtype=jnp.int32),
    )

@jax.jit
def get(dnd_buffer, query_features, k=16, temperature=0.1):    
    keys_norm = dnd_buffer.keys / (jnp.linalg.norm(dnd_buffer.keys, axis=-1, keepdims=True) + 1e-8)
    query_norm = query_features / (jnp.linalg.norm(query_features) + 1e-8)
    
    logits = jnp.dot(keys_norm, query_norm)
    
    valid_mask = dnd_buffer.age > 0
    logits = jnp.where(valid_mask, logits, -1e9)
    
    top_logits, top_indices = jax.lax.top_k(logits, k)

    top_weights = jax.nn.softmax(top_logits / temperature)
    
    top_values = jnp.take(dnd_buffer.values, top_indices, axis=0)
    values = jnp.dot(top_weights, top_values)
    
    return values

@jax.jit
def put(dnd_buffer: DND, query_features: jnp.ndarray, values: jnp.ndarray) -> DND:
    insert_idx = jnp.argmin(dnd_buffer.age)
    new_clock = dnd_buffer.clock + 1
    new_keys = dnd_buffer.keys.at[insert_idx].set(query_features)
    new_values = dnd_buffer.values.at[insert_idx].set(values)
    new_age = dnd_buffer.age.at[insert_idx].set(new_clock)

    return dnd_buffer.replace(
        keys=new_keys,
        values=new_values,
        age=new_age,
        clock=new_clock
    )

@jax.jit
def one_step_update(dnd_buffer: DND, curr_features: jnp.ndarray, action_t: int, reward_t: float, next_features: jnp.ndarray,\
    done: bool, curr_q_vals: jnp.ndarray, gamma: float = 0.99, alpha: float = 0.1, similarity_threshold: float = 0.95) -> DND:
    
    next_qs = get(dnd_buffer, next_features)
    max_next_q = jnp.max(next_qs)
    target = reward_t + gamma * max_next_q * (1.0 - done)

    keys_norm = dnd_buffer.keys / (jnp.linalg.norm(dnd_buffer.keys, axis=-1, keepdims=True) + 1e-8)
    query_norm = curr_features / (jnp.linalg.norm(curr_features) + 1e-8)
    similarities = jnp.dot(keys_norm, query_norm)
    
    valid_mask = dnd_buffer.age > 0
    similarities = jnp.where(valid_mask, similarities, -2.0)
    
    best_idx = jnp.argmax(similarities)
    max_sim = similarities[best_idx]    
    
    is_duplicate = max_sim > similarity_threshold

    def update_existing_fn(dnd):
        old_qs = dnd.values[best_idx]
        updated_q_value = old_qs[action_t] + alpha * (target - old_qs[action_t])
        new_qs = old_qs.at[action_t].set(updated_q_value)
        
        new_values = dnd.values.at[best_idx].set(new_qs)
        new_clock = dnd.clock + 1
        new_age = dnd.age.at[best_idx].set(new_clock)
        
        return dnd.replace(values=new_values, age=new_age, clock=new_clock)

    def insert_new_fn(dnd):
        new_qs = curr_q_vals.at[action_t].set(target)
        return put(dnd, curr_features, new_qs)

    updated_dnd = jax.lax.cond(
        is_duplicate, 
        update_existing_fn, 
        insert_new_fn, 
        dnd_buffer
    )
    
    return updated_dnd