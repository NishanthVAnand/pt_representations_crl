from linecache import cache
import jax
import jax.numpy as jnp
import chex
import dataclasses as dc

from hashing_jax import *

@chex.dataclass
class MinHashLSHCache:
    # Table (per-band buckets with S slots)
    tags:   jnp.ndarray   # (BANDS, NUM_BUCKETS, S) uint32
    values: jnp.ndarray   # (BANDS, NUM_BUCKETS, S, NUM_ACTIONS) float32
    valid:  jnp.ndarray   # (BANDS, NUM_BUCKETS, S) bool
    age:    jnp.ndarray   # (BANDS, NUM_BUCKETS, S) int32
    clock:  jnp.ndarray   # () int32

    # Seeds / hyperparams
    seeds_elem: jnp.ndarray  # (K,) uint32   -- K = BANDS * ROWS_PER_BAND
    seeds_band: jnp.ndarray  # (BANDS,) uint32
    seed_tag:   jnp.ndarray   # type: ignore
    bucket_mask: jnp.uint32  # type: ignore   (NUM_BUCKETS - 1), power-of-two
    rows_per_band: int
    num_bands: int
    slots_per_bucket: int
    num_buckets: int
    num_actions: int
    ctx_scale: jnp.float32   # type: ignore
    crop_mask: jnp.ndarray   # (H, W) bool  True=keep, False=exclude
    lambda_wt : jnp.float32  # type: ignore

def init_minhash_lsh_cache(
    rng: jnp.ndarray,
    *,
    num_actions: int,
    num_buckets: int = 4096,
    num_bands: int = 8,
    rows_per_band: int = 4,
    slots_per_bucket: int = 2,
    ctx_scale: float = 1024.0,
    H: int = 7,                
    W: int = 9,                
    crop_size: int = -1,
    lambda_wt: float = 0.5,
) -> MinHashLSHCache:
    """
    Initialize a MinHash LSH cache with bands and slot-LRU.
    num_buckets must be a power of two.
    """
    K = num_bands * rows_per_band

    rng, rk = jax.random.split(rng)
    seeds_elem = jax.vmap(mix2key)(jax.random.split(rk, K))          # (K,)

    rng, rb = jax.random.split(rng)
    seeds_band = jax.vmap(mix2key)(jax.random.split(rb, num_bands))  # (BANDS,)

    rng, rt = jax.random.split(rng)
    seed_tag = mix2key(rt)

    # jax.debug.print("Seeds: elem {se}, band {sb}, tag {st}", se=seeds_elem, sb=seeds_band, st=seed_tag)

    BANDS, S, A, NB = num_bands, slots_per_bucket, num_actions, num_buckets

    crop_mask = make_center_crop_mask(H, W, crop_size)  # (H,W) bool

    return MinHashLSHCache(
        tags   = jnp.zeros((BANDS, NB, S), dtype=jnp.uint32),
        values = jnp.zeros((BANDS, NB, S, A), dtype=jnp.float32),
        valid  = jnp.zeros((BANDS, NB, S), dtype=bool),
        age    = jnp.zeros((BANDS, NB, S), dtype=jnp.int32),
        clock  = jnp.array(1, dtype=jnp.int32),
        seeds_elem = seeds_elem,
        seeds_band = seeds_band,
        seed_tag   = seed_tag,
        bucket_mask = jnp.uint32(num_buckets - 1),
        rows_per_band = int(rows_per_band),
        num_bands = int(num_bands),
        slots_per_bucket = int(slots_per_bucket),
        num_buckets = int(num_buckets),
        num_actions = int(num_actions),
        ctx_scale = jnp.float32(ctx_scale),
        crop_mask = crop_mask,
        lambda_wt = jnp.float32(lambda_wt),
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
    seed_tag: jnp.uint32,    # type: ignore
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

    sig = jnp.minimum(min_tok, ctx_per_seed)  # (B,K) uint32

    # Full-signature tag
    tag = hash32_words(sig, seed_tag)  # (B,) uint32
    return sig, tag

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
def _put_one_bidx(cache: MinHashLSHCache,
                  bidx: jnp.ndarray,      # (H,) int32
                  tag: jnp.uint32) -> tuple[MinHashLSHCache, None]: # type: ignore
    H, NB, S, A = cache.values.shape
    arH = jnp.arange(H, dtype=jnp.int32)

    # Gather slices for this sample
    tags_HS  = cache.tags[arH, bidx, :]        # (H,S)
    valid_HS = cache.valid[arH, bidx, :]       # (H,S)
    vals_HSA = cache.values[arH, bidx, :, :]   # (H,S,A)

    # Hits
    match_HS   = valid_HS & (tags_HS == tag)           # tag is scalar ()/uint32
    hit_H      = match_HS.any(axis=1)
    hit_slot_H = jnp.argmax(match_HS, axis=1).astype(jnp.int32)

    # Touch hit ages
    age2 = cache.age.at[arH, bidx, hit_slot_H].set(
        jnp.where(hit_H, cache.clock, cache.age[arH, bidx, hit_slot_H])
    )

    # Miss policy: empty-first else LRU
    is_empty_HS  = ~valid_HS
    has_empty_H  = is_empty_HS.any(axis=1)
    empty_slot_H = jnp.argmax(is_empty_HS.astype(jnp.int32), axis=1)

    age_HS = age2[arH, bidx, :]
    # reserve hit slot
    arS = jnp.arange(S, dtype=jnp.int32)
    one_hot_HS = (arS[None, :] == hit_slot_H[:, None])
    reserved_HS = hit_H[:, None] & one_hot_HS

    cand_HS    = valid_HS & ~reserved_HS
    neg_inf    = jnp.int32((2**30))
    age_masked = jnp.where(cand_HS, age_HS, neg_inf)
    lru_slot_H = jnp.argmin(age_masked, axis=1)

    ins_slot_H = jnp.where(has_empty_H, empty_slot_H, lru_slot_H)
    ins_mask_H = ~hit_H

    # Mean init over other valid slots
    onehot_sel_HS = (arS[None, :] == ins_slot_H[:, None])
    valid_excl_HS = valid_HS & ~onehot_sel_HS
    sum_vals_HA   = (vals_HSA * valid_excl_HS[..., None]).sum(axis=1)
    denom_H       = jnp.maximum(1, valid_excl_HS.sum(axis=1))
    mean_vals_HA  = sum_vals_HA / denom_H[:, None]

    # Scatter winners only
    tags3 = cache.tags.at[arH, bidx, ins_slot_H].set(
        jnp.where(ins_mask_H, tag, cache.tags[arH, bidx, ins_slot_H])
    )
    valid3 = cache.valid.at[arH, bidx, ins_slot_H].set(
        cache.valid[arH, bidx, ins_slot_H] | ins_mask_H
    )
    values3 = cache.values.at[arH, bidx, ins_slot_H, :].set(
        jnp.where(ins_mask_H[:, None], mean_vals_HA, cache.values[arH, bidx, ins_slot_H, :])
    )
    age3 = age2.at[arH, bidx, ins_slot_H].set(
        jnp.where(ins_mask_H, cache.clock, age2[arH, bidx, ins_slot_H])
    )

    cache2 = dc.replace(cache, tags=tags3, valid=valid3, values=values3, age=age3)
    return cache2, None

@jax.jit
def put(cache: MinHashLSHCache, x1345: jnp.ndarray) -> MinHashLSHCache:
    tokens, mask, ctx_words = obs1345_to_tokens_and_ctx_words_with_crop(x1345, cache)
    sig, tag = _minhash_signature(tokens, mask, ctx_words, cache.seeds_elem, cache.seed_tag)  # (B,K)-> tag:(B,)
    bidx = lsh_buckets(sig, cache.seeds_band, cache.bucket_mask)                              # (B,H)

    def body(carry_cache, inp):
        bidx_i, tag_i = inp
        carry_cache, _ = _put_one_bidx(carry_cache, bidx_i, tag_i)
        return carry_cache, None

    carry, _ = jax.lax.scan(body, cache, (bidx, tag))
    carry = dc.replace(carry, clock=carry.clock + 1)
    return carry


@jax.jit
def get(cache, x1345: jnp.ndarray):
    # Hashing & bucket selection
    tokens, mask, ctx_words = obs1345_to_tokens_and_ctx_words_with_crop(x1345, cache)
    sig, tag = _minhash_signature(tokens, mask, ctx_words, cache.seeds_elem, cache.seed_tag)
    bidx = lsh_buckets(sig, cache.seeds_band, cache.bucket_mask)  # (B,H)

    # Gather table slices for the selected buckets (shape helpers)
    vals_BHSA  = _gather_BH_from_HNB(cache.values, bidx)   # (B,H,S,A)
    tags_BHS   = _gather_BH_from_HNB(cache.tags,   bidx)   # (B,H,S)
    valid_BHS  = _gather_BH_from_HNB(cache.valid,  bidx)   # (B,H,S)

    # Matches & counts
    match_BHS    = valid_BHS & (tags_BHS == tag[:, None, None])  # (B,H,S)
    has_match_BH = match_BHS.any(axis=2)                         # (B,H)
    valid_cnt_BH = valid_BHS.sum(axis=2)                         # (B,H)

    lam = jnp.asarray(cache.lambda_wt, vals_BHSA.dtype)
    one = jnp.asarray(1.0,            vals_BHSA.dtype)

    # Weights for the bands that *do* have a match:
    others_mask       = (valid_BHS & ~match_BHS).astype(vals_BHSA.dtype)
    match_mask        =  match_BHS.astype(vals_BHSA.dtype)
    denom_others_BH   = jnp.maximum(1, valid_cnt_BH - 1)                        # (B,H)
    w_if_match_BHS    = lam * match_mask + ((one - lam) / denom_others_BH[..., None]) * others_mask

    # Single-slot edge case: if exactly one valid slot and it's the match → weight 1.0 on it
    single_match_BH = has_match_BH & (valid_cnt_BH == 1)                        # (B,H)
    w_if_match_BHS  = jnp.where(single_match_BH[..., None], match_mask, w_if_match_BHS)

    # Bands with NO match should contribute nothing: zero weights there
    w_BHS = jnp.where(has_match_BH[..., None], w_if_match_BHS, 0.0)             # (B,H,S)

    # Pool within band → (B,H,A)
    band_vals_BHA = (w_BHS[..., None] * vals_BHSA).sum(axis=2)                  # (B,H,A)

    # Average ONLY over bands that had a match
    # contrib_B = has_match_BH.sum(axis=1)                                        # (B,)
    # denom_B   = jnp.maximum(1, contrib_B)                                       # safe
    # vals_BA   = band_vals_BHA.sum(axis=1) / denom_B[:, None]                    # (B,A)
    
    # Sum over bands that had a match (no averaging)
    vals_BA   = band_vals_BHA.sum(axis=1)

    # Hit if any band matched
    hit_B = has_match_BH.any(axis=1)                                            # (B,)

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
    sig, tag = _minhash_signature(tokens, mask, ctx_words, cache.seeds_elem, cache.seed_tag)
    bidx = lsh_buckets(sig, cache.seeds_band, cache.bucket_mask)  # (B,H)

    val_shape = cache.values.shape
    H, NB, S, A = val_shape
    alphaF = jnp.asarray(alpha, cache.values.dtype)
    lam    = jnp.asarray(cache.lambda_wt, cache.values.dtype)
    oneF   = jnp.asarray(1.0, cache.values.dtype)

    arH = jnp.arange(H, dtype=jnp.int32)                 # (H,)
    arS = jnp.arange(S, dtype=jnp.int32)                 # (S,)
    h_idx_HS = jnp.repeat(arH, S)                        # (H*S,)
    s_idx_HS = jnp.tile(arS, H)                          # (H*S,)

    def _gather_HS_from_HNS(arr_HNS, bidx_H):
        # arr_HNS: (H, NB, S). For each h, take arr[h, bidx_H[h], :]
        return arr_HNS[arH, bidx_H, :]  # (H,S)
    
    def _one(bidx_H, tag_s, act_s, err_s):
        # Gather table views for this sample
        tags_HS  = _gather_HS_from_HNS(cache.tags,  bidx_H)   # (H,S)
        valid_HS = _gather_HS_from_HNS(cache.valid, bidx_H)   # (H,S)

        # Matches & band participation
        match_HS    = valid_HS & (tags_HS == tag_s)           # (H,S)
        has_match_H = match_HS.any(axis=1)                    # (H,)
        valid_cnt_H = valid_HS.sum(axis=1)                    # (H,)

        # Build weights per band (only where there is a match)
        others_mask_HS     = (valid_HS & ~match_HS).astype(cache.values.dtype)
        match_mask_HS      =  match_HS.astype(cache.values.dtype)
        denom_others_H     = jnp.maximum(1, valid_cnt_H - 1)                  # (H,)
        w_if_match_HS      = lam * match_mask_HS + ((oneF - lam) / denom_others_H[:, None]) * others_mask_HS

        # Single-slot edge: exactly one valid & it's the match -> weight 1.0 on that slot
        single_match_H     = has_match_H & (valid_cnt_H == 1)
        w_if_match_HS      = jnp.where(single_match_H[:, None], match_mask_HS, w_if_match_HS)

        # Zero weights for bands with no match
        w_HS = jnp.where(has_match_H[:, None], w_if_match_HS, 0.0)            # (H,S)

        # Split error equally among contributing bands
        contrib = has_match_H.sum()
        band_share = (err_s.astype(cache.values.dtype) / jnp.maximum(1, contrib))

        # Per-slot delta for chosen action (no A broadcast!)
        delta_HS = alphaF * band_share * w_HS                                  # (H,S)

        # Flatten per-sample indices/updates
        b_idx_HS = jnp.repeat(bidx_H, S)                                       # (H*S,)
        a_idx_HS = jnp.full((H*S,), act_s, dtype=jnp.int32)                    # (H*S,)
        d_flat   = delta_HS.reshape(-1)                                        # (H*S,)

        return h_idx_HS, b_idx_HS, s_idx_HS, a_idx_HS, d_flat

    h_all, b_all, s_all, a_all, d_all = jax.vmap(_one, in_axes=(0, 0, 0, 0))(
        bidx, tag, action, td_err
    )
    # Concatenate batch: shape (B*H*S,)
    h_all = h_all.reshape(-1)
    b_all = b_all.reshape(-1)
    s_all = s_all.reshape(-1)
    a_all = a_all.reshape(-1)
    d_all = d_all.reshape(-1)

    values2 = cache.values.at[h_all, b_all, s_all, a_all].add(d_all)

    return dc.replace(cache, values=values2)

@jax.jit
def decay_values(cache: MinHashLSHCache, decay: float) -> MinHashLSHCache:
    return dc.replace(cache, values=cache.values * jnp.asarray(decay, cache.values.dtype))