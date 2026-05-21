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
def batched_update(dnd_buffer: DND, batched_curr_features: jnp.ndarray, batched_actions: jnp.ndarray, batched_rewards: jnp.ndarray,\
                                    batched_next_features: jnp.ndarray, batched_dones: jnp.ndarray, batched_curr_q_vals: jnp.ndarray, gamma: float = 0.99,\
                                    alpha: float = 0.1, sim_threshold: float = 0.95) -> DND:
    
    batch_size = batched_curr_features.shape[0]

    batched_get = jax.vmap(get, in_axes=(None, 0))
    next_qs = batched_get(dnd_buffer, batched_next_features)
    max_next_q = jnp.max(next_qs, axis=-1)
    targets = batched_rewards + gamma * max_next_q * (1.0 - batched_dones)

    # similarities
    keys_norm = dnd_buffer.keys / (jnp.linalg.norm(dnd_buffer.keys, axis=-1, keepdims=True) + 1e-8)
    query_norm = batched_curr_features / (jnp.linalg.norm(batched_curr_features, axis=-1, keepdims=True) + 1e-8)
    
    sim = jnp.dot(query_norm, keys_norm.T) 
    valid_mask = dnd_buffer.age > 0
    sim_masked = jnp.where(valid_mask[None, :], sim, -2.0) # lowest similarity for empty slots

    best_match_indices = jnp.argmax(sim_masked, axis=-1) 
    max_sims = jnp.max(sim_masked, axis=-1)              
    is_duplicate = max_sims > sim_threshold # it is a match only if sim score is > threshold              

    old_qs = dnd_buffer.values[best_match_indices]       
    batch_idx = jnp.arange(batch_size)
    taken_qs = old_qs[batch_idx, batched_actions]
    updated_taken_qs = taken_qs + alpha * (targets - taken_qs) # updates q-learning
    
    new_action_qs = jnp.where(is_duplicate, updated_taken_qs, targets)

    _, oldest_indices = jax.lax.top_k(-dnd_buffer.age, batch_size) 
    insert_ranks = jnp.cumsum(~is_duplicate) - 1
    packed_oldest_indices = oldest_indices[insert_ranks] # oldest slot to the first batch element that is not present in DND and so on
    
    final_indices = jnp.where(is_duplicate, best_match_indices, packed_oldest_indices)

    row_bases = jnp.where(is_duplicate[:, None], old_qs, batched_curr_q_vals)
    
    base_canvas = dnd_buffer.values.at[final_indices].set(row_bases)

    summed_qs = jnp.zeros_like(dnd_buffer.values).at[final_indices, batched_actions].add(new_action_qs)
    counts = jnp.zeros_like(dnd_buffer.values).at[final_indices, batched_actions].add(1)

    mean_qs = summed_qs / jnp.maximum(counts, 1)
    
    was_hit = counts > 0 
    
    new_dnd_values = jnp.where(was_hit, mean_qs, base_canvas)

    old_keys = dnd_buffer.keys[best_match_indices]
    final_keys = jnp.where(is_duplicate[:, None], old_keys, batched_curr_features)

    new_clock = dnd_buffer.clock + 1
    new_dnd_keys = dnd_buffer.keys.at[final_indices].set(final_keys)
    new_dnd_age = dnd_buffer.age.at[final_indices].set(new_clock)

    return dnd_buffer.replace(
        keys=new_dnd_keys,
        values=new_dnd_values,
        age=new_dnd_age,
        clock=new_clock
    )