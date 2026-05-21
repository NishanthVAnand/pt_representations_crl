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

def hash32_words(words_u32: jnp.ndarray, seed: jnp.uint32) -> jnp.ndarray:  # type: ignore
    """
    Hash a list of u32 words with fmix32+xor reduce.
    words_u32: (..., W) uint32 -> (...,) uint32
    """
    mixed = fmix32(words_u32 ^ seed)
    return jnp.bitwise_xor.reduce(mixed, axis=-1)

def ones_and_zeros_to_tokens(x_vec: jnp.ndarray) -> jnp.ndarray:
    *lead, L = x_vec.shape
    flat = x_vec.reshape(-1, L)  # (BATCH, L)
    B = flat.shape[0]

    b_vecs_base = jnp.arange(L, dtype=jnp.uint32)[None, :]
    b_vecs = jnp.broadcast_to(b_vecs_base, (B, L))
    words = jnp.stack([b_vecs, flat.astype(jnp.uint32)], axis=-1)  # (BATCH, L, 2)
    tokens = hash32_words(words, jnp.uint32(0))
    tokens = tokens.reshape(*lead, L)

    return tokens

def lsh_buckets(sig: jnp.ndarray, cache) -> jnp.ndarray:
    NT = cache.seeds_band.shape[0]
    sigBR = sig.reshape(sig.shape[0], NT, -1)  # (B, TABLES, R)

    # hash each band's rows -> (B,)
    def band_hash(wordsBR, seed):
        return hash32_words(wordsBR, seed)     # (B,)

    bh = jax.vmap(band_hash, in_axes=(1, 0))(sigBR, cache.seeds_band)  # (TABLES, B)
    bh = bh.T                                                    # (B, TABLES)
    return (bh & cache.bucket_mask).astype(jnp.int32) 