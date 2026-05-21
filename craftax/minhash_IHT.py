from linecache import cache
import jax
import jax.numpy as jnp
import chex
import dataclasses as dc

from hashing_jax import *

@chex.dataclass
class MinHashLSHCache:
    # Table (per-band buckets with S slots)
    values: jnp.ndarray   # (BANDS, NUM_BUCKETS, NUM_ACTIONS) float32

    # Seeds / hyperparams
    seeds_elem: jnp.ndarray  # (K,) uint32   -- K = BANDS * ROWS_PER_BAND
    seeds_band: jnp.ndarray  # (BANDS,) uint32
    bucket_mask: jnp.uint32  # type: ignore   (NUM_BUCKETS - 1), power-of-two
    rows_per_band: int
    num_bands: int
    num_buckets: int
    num_actions: int
    ctx_scale: jnp.float32   # type: ignore
    crop_mask: jnp.ndarray   # (H, W) bool  True=keep, False=exclude

def init_minhash_IHT_cache(
    rng: jnp.ndarray,
    *,
    num_bands: int = 8,
    num_actions: int,
    num_buckets: int = 4096,
    rows_per_band: int = 4,
    ctx_scale: float = 1024.0,
    H: int = 7,                
    W: int = 9,                
    crop_size: int = -1,
) -> MinHashLSHCache:
    """
    Initialize a MinHash IHT cache.
    num_buckets must be a power of two.
    """
    K = num_bands * rows_per_band

    rng, rk = jax.random.split(rng)
    seeds_elem = jax.vmap(mix2key)(jax.random.split(rk, K))          # (K,)

    rng, rb = jax.random.split(rng)
    seeds_band = jax.vmap(mix2key)(jax.random.split(rb, num_bands))  # (BANDS,)

    BANDS, A, NB = num_bands, num_actions, num_buckets

    crop_mask = make_center_crop_mask(H, W, crop_size)  # (H,W) bool

    return MinHashLSHCache(
        values = jnp.zeros((BANDS, NB, A), dtype=jnp.float32),
        seeds_band = seeds_band,
        bucket_mask = jnp.uint32(num_buckets - 1),
        seeds_elem = seeds_elem,
        rows_per_band = int(rows_per_band),
        num_bands = int(num_bands),
        num_buckets = int(num_buckets),
        num_actions = int(num_actions),
        ctx_scale = jnp.float32(ctx_scale),
        crop_mask = crop_mask,
    )

def obs1345_to_tokens_and_ctx_words_with_crop(
    x1345: jnp.ndarray,      # (B, 1345) float32 in [0,1]
    cache: MinHashLSHCache,  # to access crop_mask and ctx_scale  # type: ignore
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      tokens_u32: (B, 2*H*W + 2) uint32  (grid1 + grid2 + direction + sleep)
      mask:       (B, 2*H*W + 2) bool
      ctx_words:  (B, Wc) uint32         (packed ctx scalars; sleep excluded)
    """
    B = x1345.shape[0]
    H, W = cache.crop_mask.shape
    HW = H * W

    # Split the 1345 vector into parts
    x = x1345.astype(jnp.float32)
    img1_flat = x[:, :1071]                     # (B, 7*9*17)
    img2_flat = x[:, 1071:1323]                 # (B, 7*9*4)
    ctx_a = x[:, 1323:1335]                     # (B, 12)
    dir4   = x[:, 1339:1343]                    # (B, 4) one-hot direction
    sleep1 = x[:, 1344:1345]                    # (B, 1) binary {0,1}
    ctx_b  = x[:, 1335:1339]                    # (B, 4)
    ctx_c = x[:, 1343:1344]

    # Reshape grids
    grid1 = img1_flat.reshape(B, H, W, 17)      # (B, H, W, 17)
    grid2 = img2_flat.reshape(B, H, W, 4)       # (B, H, W, 4)

    # Promote grid2 to 5 channels by adding a leading zero channel
    zeros = jnp.zeros((B, H, W, 1), dtype=grid2.dtype)
    grid2_5 = jnp.concatenate([zeros, grid2], axis=-1)  # (B,H,W,5)

    # Apply fixed crop mask (broadcast on channel)
    cm = cache.crop_mask[None, :, :, None]      # (1,H,W,1)
    grid1_c = grid1 * cm                        # (B,H,W,17)
    grid2_c = grid2_5 * cm                      # (B,H,W,5)

    # Convert grids to per-cell tokens + masks
    t1, m1 = grid_tokens_from_onehot(grid1_c, src_id=0)   # (B,HW), (B,HW)
    t2, m2 = grid_tokens_from_onehot(grid2_c, src_id=1)   # (B,HW), (B,HW)

    # Direction → single token (one of 4 bits on)
    dir_idx = jnp.argmax(dir4, axis=-1).astype(jnp.uint32)         # (B,)
    dir_tok = fmix32(dir_idx ^ jnp.uint32(0xA5A5A5A5))             # (B,) uint32
    t_dir = dir_tok[:, None]                                       # (B,1)
    m_dir = (dir4.sum(axis=-1) > 0)[:, None]                       # (B,1)

    # Sleep → single token (always present; mask=True)
    sleep_u32 = sleep1.squeeze(-1).round().astype(jnp.uint32)      # (B,)
    slp_tok = fmix32(sleep_u32 ^ jnp.uint32(0xC0FFEE01))           # (B,)
    t_slp = slp_tok[:, None]                                       # (B,1)
    m_slp = jnp.ones((B, 1), dtype=bool)                           # (B,1)

    # Concatenate tokens and masks  -> length = 2*HW + 2
    tokens = jnp.concatenate([t1, t2, t_dir, t_slp], axis=1)       # (B, 2*HW+2)
    mask   = jnp.concatenate([m1, m2, m_dir, m_slp], axis=1)       # (B, 2*HW+2)

    # Context words = [ctx_a (12), ctx_b (4)]  (sleep excluded)
    ctx_vec = jnp.concatenate([ctx_a, ctx_b, ctx_c], axis=-1)             # (B,16)
    ctx_i16 = quantize_i16_01(ctx_vec, scale=cache.ctx_scale)      # (B,16) i16
    ctx_words = pack_i16_to_u32(ctx_i16)                           # (B, 8) u32

    return tokens.astype(jnp.uint32), mask, ctx_words

def _minhash_signature(
    tokens: jnp.ndarray,     # (B, M) uint32
    mask:   jnp.ndarray,     # (B, M) bool
    ctx_words: jnp.ndarray,  # (B, Wc) uint32
    seeds_elem: jnp.ndarray, # (K,) uint32
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute MinHash signature (B,K) and 32-bit tag (B,) from tokens + ctx.
    """
    B, M = tokens.shape
    K = seeds_elem.shape[0]
    big = jnp.uint32(0xFFFFFFFF)

    # Hash tokens with each seed, take min across elements
    tok_h = fmix32(tokens[:, None, :] ^ seeds_elem[None, :, None])  # (B,K,M)
    tok_h = jnp.where(mask[:, None, :], tok_h, big)
    min_tok = tok_h.min(axis=-1).astype(jnp.uint32)                 # (B,K)

    # Context contribution: per-seed hash over ctx words, then min with token-hash
    ctx_per_seed = jax.vmap(lambda s: hash32_words(ctx_words, s))(seeds_elem)  # (K,B)
    ctx_per_seed = ctx_per_seed.T.astype(jnp.uint32)                             # (B,K)

    # # ctx_words: (B,Wc), seeds_elem: (K,)
    # ctx_b = jnp.broadcast_to(ctx_words[:, None, :], (ctx_words.shape[0], seeds_elem.shape[0], ctx_words.shape[1]))
    # ctx_per_seed = _hash32_words(ctx_b, seeds_elem[None, :, None]).astype(jnp.uint32)  # (B,K)

    sig = jnp.minimum(min_tok, ctx_per_seed)  # (B,K) uint32
    return sig

def _gather_BH_from_HNB(arr_HNB: jnp.ndarray, bidx_BH: jnp.ndarray) -> jnp.ndarray:
    """
    arr_HNB: (H, NB)
    bidx_BH: (B, H) int32 bucket indices
    returns: (B, H, ...) — slice arr_HNB[h, bidx[b,h], ...]
    """
    H = arr_HNB.shape[0]

    def gather_one_b(bidx_h: jnp.ndarray) -> jnp.ndarray:
        # bidx_h: (H,)
        return arr_HNB[jnp.arange(H), bidx_h]  # (H, ...)

    return jax.vmap(gather_one_b)(bidx_BH)  # (B,H, ...)


@jax.jit
def get(cache, x1345: jnp.ndarray):
    # Hashing & bucket selection
    tokens, mask, ctx_words = obs1345_to_tokens_and_ctx_words_with_crop(x1345, cache)
    sig = _minhash_signature(tokens, mask, ctx_words, cache.seeds_elem)
    bidx = lsh_buckets(sig, cache.seeds_band, cache.bucket_mask)  # (B,H)

    # Gather table slices for the selected buckets (shape helpers)
    vals_BHSA  = _gather_BH_from_HNB(cache.values, bidx)   # (B,H,A)
    
    # Sum over bands
    vals_BA   = vals_BHSA.sum(axis=1)

    # Hit always set to true
    hit_B = jnp.ones_like(vals_BA[:, 0], dtype=bool)

    return vals_BA, hit_B

@jax.jit
def td_update_hit_only(
    cache: MinHashLSHCache,
    x1345: jnp.ndarray,       # (B, 1345)
    action: jnp.ndarray,      # (B,) int32
    td_err: jnp.ndarray,      # (B,) float32
    alpha: float,             # scalar
) -> MinHashLSHCache:
    tokens, mask, ctx_words = obs1345_to_tokens_and_ctx_words_with_crop(x1345, cache)
    sig = _minhash_signature(tokens, mask, ctx_words, cache.seeds_elem)
    bidx = lsh_buckets(sig, cache.seeds_band, cache.bucket_mask)  # (B,H)

    val_shape = cache.values.shape
    H, NB, A = val_shape
    alphaF = jnp.asarray(alpha/H, cache.values.dtype)

    def update_one(bidx_h: jnp.ndarray,
                       action_b: jnp.ndarray,
                       td_err_b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

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