import jax
import jax.numpy as jnp
import chex
import dataclasses as dc
import dm_pix as pix

from hashing_jax import *

@chex.dataclass
class MinHashLSHCache:
    tags:   jnp.ndarray   # (TABLES, BINS, SLOTS) uint32
    values: jnp.ndarray   # (TABLES, BINS, SLOTS, NUM_ACTIONS) float32
    valid:  jnp.ndarray   # (TABLES, BINS, SLOTS) bool
    age:    jnp.ndarray   # (TABLES, BINS, SLOTS) int32
    clock:  jnp.ndarray   # () int32

    seeds_elem: jnp.ndarray  # (K,) uint32, K = TABLES * ROWS_PER_BAND
    seeds_band: jnp.ndarray  # (TABLES,) uint32
    seed_tag:   jnp.ndarray   # type: ignore
    bucket_mask: jnp.uint32  # type: ignore   (NUM_BINS - 1), power-of-two
    rows_per_band: int
    num_tables: int
    slots_per_bin: int
    num_bins: int
    num_actions: int
    levels: int
    lambda_wt : jnp.float32  # type: ignore

def init_minhash_lsh_cache(rng: jnp.ndarray, *, num_actions: int, num_bins: int = 4096, num_tables: int = 8, 
                           rows_per_band: int = 4, slots_per_bin: int = 2, lambda_wt: float = 0.5, levels: int = 32) -> MinHashLSHCache:
    """
    Initialize a MinHash LSH cache with bands and slot-LRU.
    num_buckets must be a power of two.
    """
    K = num_tables * rows_per_band

    rng, rk = jax.random.split(rng)
    seeds_elem = jax.vmap(mix2key)(jax.random.split(rk, K))          # (K,)

    rng, rb = jax.random.split(rng)
    seeds_band = jax.vmap(mix2key)(jax.random.split(rb, num_tables))  # (tables,)

    rng, rt = jax.random.split(rng)
    seed_tag = mix2key(rt)

    NT, S, A, NB = num_tables, slots_per_bin, num_actions, num_bins

    return MinHashLSHCache(
        tags   = jnp.zeros((NT, NB, S), dtype=jnp.uint32),
        values = jnp.zeros((NT, NB, S, A), dtype=jnp.float32),
        valid  = jnp.zeros((NT, NB, S), dtype=bool),
        age    = jnp.zeros((NT, NB, S), dtype=jnp.int32),
        clock  = jnp.array(1, dtype=jnp.int32),
        seeds_elem = seeds_elem,
        seeds_band = seeds_band,
        seed_tag   = seed_tag,
        bucket_mask = jnp.uint32(num_bins - 1),
        rows_per_band = int(rows_per_band),
        num_tables = int(num_tables),
        slots_per_bin = int(slots_per_bin),
        num_bins = int(num_bins),
        num_actions = int(num_actions),
        lambda_wt = jnp.float32(lambda_wt),
        levels = int(levels)
    )

def _img_tokens_to_min_hash_signature(x_img_tokens: jnp.ndarray, cache: MinHashLSHCache) -> tuple[jnp.ndarray, jnp.ndarray]:
    mask = x_img_tokens > 0
    tokens = ones_and_zeros_to_tokens(x_img_tokens)

    hash_array = fmix32(tokens[:, None, :] ^ cache.seeds_elem[None, :, None])  # (B,K,M)
    hash_array = jnp.where(mask[:, None, :], hash_array, jnp.uint32(0xFFFFFFFF))

    min_hash = hash_array.min(axis=-1).astype(jnp.uint32)  # (B,K)

    tag = hash32_words(min_hash, cache.seed_tag)  # (B,) uint32
    
    return min_hash, tag

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
def _put_one_bidx(cache: MinHashLSHCache, bidx: jnp.ndarray,  tag: jnp.uint32) -> tuple[MinHashLSHCache, None]: # type: ignore
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
def put(cache: MinHashLSHCache, rgb_images: jnp.ndarray) -> MinHashLSHCache:
    img_tokens = _rgb_tokens(rgb_images, cache.levels)
    sig, tag = _img_tokens_to_min_hash_signature(img_tokens, cache)
    bidx = lsh_buckets(sig, cache)

    def body(carry_cache, inp):
        bidx_i, tag_i = inp
        carry_cache, _ = _put_one_bidx(carry_cache, bidx_i, tag_i)
        return carry_cache, None

    carry, _ = jax.lax.scan(body, cache, (bidx, tag))
    carry = dc.replace(carry, clock=carry.clock + 1)
    return carry

@jax.jit
def get(cache, rgb_images: jnp.ndarray):
    # Hashing & bucket selection
    img_tokens = _rgb_tokens(rgb_images, cache.levels)
    sig, tag = _img_tokens_to_min_hash_signature(img_tokens, cache)
    bidx = lsh_buckets(sig, cache)

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
    #   λ on matching slot, spread (1-λ) across other valid slots in that band.
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
    vals_BA   = band_vals_BHA.sum(axis=1)

    # Hit if any band matched
    hit_B = has_match_BH.any(axis=1)                                            # (B,)

    return vals_BA, hit_B

@jax.jit
def td_update_hit_only(cache: MinHashLSHCache, rgb_images: jnp.ndarray, action: jnp.ndarray,  td_err: jnp.ndarray,  alpha: float) -> MinHashLSHCache:
    
    img_tokens = _rgb_tokens(rgb_images, cache.levels)
    sig, tag = _img_tokens_to_min_hash_signature(img_tokens, cache)
    bidx = lsh_buckets(sig, cache)

    val_shape = cache.values.shape
    H, NB, S, A = val_shape
    alphaF = jnp.asarray(alpha, cache.values.dtype)
    lam    = jnp.asarray(cache.lambda_wt, cache.values.dtype)
    oneF   = jnp.asarray(1.0, cache.values.dtype)

    arH = jnp.arange(H, dtype=jnp.int32)                 # (H,)
    arS = jnp.arange(S, dtype=jnp.int32)                 # (S,)
    h_idx_HS = jnp.repeat(arH, S)                        # (H*S,)
    s_idx_HS = jnp.tile(arS, H)                          # (H*S,)

    # --- small helper: gather (H,S) slice from (H,NB,S) using per-band buckets ---
    def _gather_HS_from_HNS(arr_HNS, bidx_H):
        # arr_HNS: (H, NB, S). For each h, take arr[h, bidx_H[h], :]
        return arr_HNS[arH, bidx_H, :]  # (H,S)

    # ---------- per-sample update (vmapped over B) ----------
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