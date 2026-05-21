import jax
import jax.numpy as jnp

def fmix32(x: jnp.ndarray) -> jnp.ndarray:
    """x: (...,) uint32 -> (...,) uint32"""
    x ^= (x >> jnp.uint32(16))
    x *= jnp.uint32(0x85EBCA6B)
    x ^= (x >> jnp.uint32(13))
    x *= jnp.uint32(0xC2B2AE35)
    x ^= (x >> jnp.uint32(16))
    return x

def mix2key(k):
    # take a PRNG key -> two u32 -> avalanche -> one u32
    k1, k2 = jax.random.split(k)
    a = jax.random.bits(k1, shape=(), dtype=jnp.uint32)
    b = jax.random.bits(k2, shape=(), dtype=jnp.uint32)
    return jnp.bitwise_xor(fmix32(a), fmix32(b))

def make_center_crop_mask(H: int, W: int, crop_size: int) -> jnp.ndarray:
    hs = jnp.arange(H, dtype=jnp.int32)[:, None]  # (H,1)
    ws = jnp.arange(W, dtype=jnp.int32)[None, :]  # (1,W)

    full_mask = jnp.ones((H, W), dtype=bool)

    max_c = jnp.minimum(jnp.int32(H), jnp.int32(W))
    ce = jnp.clip(jnp.int32(crop_size), 1, max_c)

    ch, cw = jnp.int32(H // 2), jnp.int32(W // 2)
    half   = ce // 2
    rmin, cmin = ch - half, cw - half
    rmax, cmax = rmin + ce - 1, cmin + ce - 1

    row_ok = (hs >= rmin) & (hs <= rmax)
    col_ok = (ws >= cmin) & (ws <= cmax)
    crop_mask = (row_ok & col_ok)

    return jnp.where(crop_size < 0, full_mask, crop_mask)

def hash32_words(words_u32: jnp.ndarray, seed: jnp.uint32) -> jnp.ndarray:  # type: ignore
    """
    Hash a list of u32 words with fmix32+xor reduce.
    words_u32: (..., W) uint32 -> (...,) uint32
    """
    mixed = fmix32(words_u32 ^ seed)
    return jnp.bitwise_xor.reduce(mixed, axis=-1)

def grid_tokens_from_onehot(
    grid: jnp.ndarray,   # (..., H, W, C)
    src_id: int,         # distinguish which grid
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Turn a one-hot HxWxC grid (one object per cell, may be empty) into per-cell tokens.
    Returns:
      tokens_u32: (..., H*W) uint32   (arbitrary for empties; masked out)
      mask:       (..., H*W) bool     (True where object present)
    """
    *lead, H, W, C = grid.shape
    flat = grid.reshape(-1, H, W, C)
    B = flat.shape[0]

    present = (flat.sum(axis=-1) > 0)                 # (B,H,W)
    cls = jnp.argmax(flat, axis=-1).astype(jnp.uint32)  # (B,H,W)

    rows = jnp.arange(H, dtype=jnp.uint32)[None, :, None]
    cols = jnp.arange(W, dtype=jnp.uint32)[None, None, :]
    rows = jnp.broadcast_to(rows, (B, H, W))
    cols = jnp.broadcast_to(cols, (B, H, W))
    src  = jnp.uint32(src_id) * jnp.ones_like(rows, dtype=jnp.uint32)

    words = jnp.stack([rows, cols, cls, src], axis=-1)     # (B,H,W,4)
    words = words.reshape(B, H*W, 4)                       # (B,HW,4)
    tokens = hash32_words(words, jnp.uint32(0))           # (B,HW)
    tokens = tokens.reshape(*lead, H*W)
    mask   = present.reshape(*lead, H*W)
    return tokens, mask

def quantize_i16_01(x: jnp.ndarray, scale: float = 1024.0) -> jnp.ndarray:
    """
    x in [0,1] -> int16. 0 -> 0; 1 -> ~scale.
    """
    y = jnp.clip(x, 0.0, 1.0)
    y = jnp.round(y * jnp.asarray(scale, y.dtype))
    return y.astype(jnp.int16)

def pack_i16_to_u32(v_i16: jnp.ndarray) -> jnp.ndarray:
    """
    Strategy:
      - Split into even and odd index streams.
      - Always append one zero to the odd stream, then slice to match the
        even stream length (which is ceil(D/2)). This avoids any conditional pad.
    """
    v_u32 = (v_i16.astype(jnp.uint32) & jnp.uint32(0xFFFF))  # (..., D)
    v_even = v_u32[..., 0::2]                                 # (..., ceil(D/2))
    v_odd_base = v_u32[..., 1::2]                             # (..., floor(D/2))

    # Always append one zero, then slice back to v_even length (static).
    # Safe even when D is even: we just drop the extra zero.
    zero_tail = jnp.zeros_like(v_u32[..., -1:])
    v_odd_padded = jnp.concatenate([v_odd_base, zero_tail], axis=-1)
    v_odd = v_odd_padded[..., :v_even.shape[-1]]              # (..., ceil(D/2))

    packed = (v_even | (v_odd << jnp.uint32(16))).astype(jnp.uint32)  # (..., ceil(D/2))
    return packed

def lsh_buckets(
    sig: jnp.ndarray,         # (B, K) uint32
    seeds_band: jnp.ndarray,  # (BANDS,) uint32
    bucket_mask: jnp.uint32,  # type: ignore # () uint32
) -> jnp.ndarray:
    """
    Turn K-length signature into per-band bucket indices.
    Returns: (B, BANDS) int32
    """
    BANDS = seeds_band.shape[0]                 # static
    # K must be divisible by BANDS; let reshape infer rows_per_band
    sigBR = sig.reshape(sig.shape[0], BANDS, -1)  # (B, BANDS, R)

    # hash each band's rows -> (B,)
    def band_hash(wordsBR, seed):
        return hash32_words(wordsBR, seed)     # (B,)

    bh = jax.vmap(band_hash, in_axes=(1, 0))(sigBR, seeds_band)  # (BANDS, B)
    bh = bh.T                                                    # (B, BANDS)
    return (bh & bucket_mask).astype(jnp.int32)  