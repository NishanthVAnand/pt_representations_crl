import jax
import jax.numpy as jnp
import chex
import dataclasses as dc
import dm_pix as pix

from hashing_jax import *

@chex.dataclass
class MinHashLSHCache:
    values: jnp.ndarray   # (TABLES, BINS, NUM_ACTIONS) float32

    seeds_elem: jnp.ndarray  # (K,) uint32, K = TABLES * ROWS_PER_BAND
    seeds_band: jnp.ndarray  # (TABLES,) uint32
    bucket_mask: jnp.uint32  # type: ignore   (NUM_BINS - 1), power-of-two
    rows_per_band: int
    num_tables: int
    num_bins: int
    num_actions: int
    levels: int
def init_minhash_lsh_cache(rng: jnp.ndarray, *, num_actions: int, num_bins: int = 4096, num_tables: int = 8, 
                           rows_per_band: int = 4, levels: int = 32) -> MinHashLSHCache:
    """
    Initialize a MinHash LSH cache with bands and slot-LRU.
    num_buckets must be a power of two.
    """
    K = num_tables * rows_per_band

    rng, rk = jax.random.split(rng)
    seeds_elem = jax.vmap(mix2key)(jax.random.split(rk, K))          # (K,)

    rng, rb = jax.random.split(rng)
    seeds_band = jax.vmap(mix2key)(jax.random.split(rb, num_tables))  # (tables,)

    NT, A, NB = num_tables, num_actions, num_bins

    return MinHashLSHCache(
        values = jnp.zeros((NT, NB, A), dtype=jnp.float32),
        seeds_elem = seeds_elem,
        seeds_band = seeds_band,
        bucket_mask = jnp.uint32(num_bins - 1),
        rows_per_band = int(rows_per_band),
        num_tables = int(num_tables),
        num_bins = int(num_bins),
        num_actions = int(num_actions),
        levels = int(levels)
    )

def _ones_and_zeros_to_tokens(x_vec: jnp.ndarray) -> jnp.ndarray:
    *lead, L = x_vec.shape
    flat = x_vec.reshape(-1, L)  # (BATCH, L)
    B = flat.shape[0]

    b_vecs_base = jnp.arange(L, dtype=jnp.uint32)[None, :]
    b_vecs = jnp.broadcast_to(b_vecs_base, (B, L))
    words = jnp.stack([b_vecs, flat.astype(jnp.uint32)], axis=-1)  # (BATCH, L, 2)
    tokens = hash32_words(words, jnp.uint32(0))
    tokens = tokens.reshape(*lead, L)

    return tokens

def _img_tokens_to_min_hash_signature(x_img_tokens: jnp.ndarray, cache: MinHashLSHCache) -> tuple[jnp.ndarray, jnp.ndarray]:
    mask = x_img_tokens > 0
    tokens = _ones_and_zeros_to_tokens(x_img_tokens)

    hash_array = fmix32(tokens[:, None, :] ^ cache.seeds_elem[None, :, None])  # (B,K,M)
    hash_array = jnp.where(mask[:, None, :], hash_array, jnp.uint32(0xFFFFFFFF))

    min_hash = hash_array.min(axis=-1).astype(jnp.uint32)  # (B,K)
    
    return min_hash

def _gather_BH_from_HNB(arr_HNB: jnp.ndarray, bidx_BH: jnp.ndarray) -> jnp.ndarray:
    """
    arr_HNB: (H, NB, S[,...])
    bidx_BH: (B, H) int32 bucket indices
    returns: (B, H, S[,...]) — slice arr_HNB[h, bidx[b,h], ...]
    """
    H = arr_HNB.shape[0]

    def gather_one_b(bidx_h: jnp.ndarray) -> jnp.ndarray:
        # bidx_h: (H,)
        return arr_HNB[jnp.arange(H), bidx_h, ...]  # (H, S[,...])

    return jax.vmap(gather_one_b)(bidx_BH)  # (B,H,S[,...])

@jax.jit
def _rgb_tokens(arr_image: jnp.ndarray, num_levels: int) -> jnp.ndarray:
    """
    Converts an RGB image to a sequence of tokens.
    """
    gray_image = pix.rgb_to_grayscale(arr_image)
    gray_image = jnp.squeeze(gray_image, axis=-1)
    
    *lead, H, W = gray_image.shape
    flat_gray = gray_image.reshape(-1, H, W)
    B = flat_gray.shape[0]

    indices = jnp.round(flat_gray * (num_levels - 1)).astype(jnp.int32)
    clss = jnp.clip(indices, 0, num_levels - 1).astype(jnp.uint32)  # Shape: (B, H, W)

    y_coords = jnp.arange(H, dtype=jnp.uint32)[None, :, None]
    x_coords = jnp.arange(W, dtype=jnp.uint32)[None, None, :]
    
    rows = jnp.broadcast_to(y_coords, (B, H, W))
    cols = jnp.broadcast_to(x_coords, (B, H, W))
    
    words = jnp.stack([rows, cols, clss], axis=-1)   # Shape: (B, H, W, 3)
    words = words.reshape(B, H * W, 3)               # Shape: (B, H*W, 3)
    
    tokens = hash32_words(words, jnp.uint32(0))     # Shape: (B, H*W)
    return tokens.reshape(*lead, H * W)

@jax.jit
def get(cache, rgb_images: jnp.ndarray):
    # Hashing & bucket selection
    img_tokens = _rgb_tokens(rgb_images, cache.levels)
    sig = _img_tokens_to_min_hash_signature(img_tokens, cache)
    bidx = lsh_buckets(sig, cache)

    # Gather table slices for the selected buckets (shape helpers)
    vals_BHA  = _gather_BH_from_HNB(cache.values, bidx)   # (B,H,A)
    vals_BA   = vals_BHA.sum(axis=1)

    # Hit if any band matched
    hit_B = jnp.ones_like(vals_BA[:, 0], dtype=bool)
    return vals_BA, hit_B

@jax.jit
def td_update_hit_only(cache: MinHashLSHCache, rgb_images: jnp.ndarray, action: jnp.ndarray,  td_err: jnp.ndarray,  alpha: float) -> MinHashLSHCache:
    img_tokens = _rgb_tokens(rgb_images, cache.levels)
    sig = _img_tokens_to_min_hash_signature(img_tokens, cache)
    bidx = lsh_buckets(sig, cache)

    val_shape = cache.values.shape
    H, NB, A = val_shape
    alphaF = jnp.asarray(alpha/H, cache.values.dtype)
    
    def update_one(bidx_h: jnp.ndarray, action_b: jnp.ndarray, td_err_b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        a_all = jnp.full((H,), action_b, dtype=jnp.int32)               # (H,)
        b_all = bidx_h                                                  # (H,)
        h_all = jnp.arange(H, dtype=jnp.int32)                          # (H,)
        d_all = alphaF * td_err_b * jnp.ones((H,), dtype=cache.values.dtype)  # (H,)

        return h_all, b_all, a_all, d_all

    h_all, b_all, a_all, d_all = jax.vmap(update_one)(bidx, action, td_err)
    
    values2 = cache.values.at[h_all, b_all, a_all].add(d_all)

    return dc.replace(cache, values=values2)

@jax.jit
def decay_values(cache: MinHashLSHCache, decay: float) -> MinHashLSHCache:
    return dc.replace(cache, values=cache.values * jnp.asarray(decay, cache.values.dtype))