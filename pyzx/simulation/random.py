import jax
import jax.numpy as jnp

key = jax.random.key(0)

probs = jnp.array(
    [
        0.05,
        0.1,
        0.02,
        0.03,
        0.2,
        0.1,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.1,
        0.02,
        0.01,
        0.00,
    ]
)
probs /= probs.sum()

logits = jnp.log(probs)

N = 10000
samples = jax.random.categorical(key, logits, shape=(N,))

# optional: unpack to 4 bits
bits = ((samples[:, None] >> jnp.arange(4)) & 1).astype(jnp.uint8)

# print(bits)


def pauli_channel_1(px, py, pz, num_samples: int = 1):
    # 00, 10, 01, 11
    probs = jnp.array([1 - px - py - pz, pz, px, py])
    logits = jnp.log(probs)
    samples = jax.random.categorical(key, logits, shape=(num_samples,))
    bits = ((samples[:, None] >> jnp.arange(2)) & 1).astype(jnp.uint8)
    return bits


def pauli_channel_2(
    pix=0,
    piy=0,
    piz=0,
    pxi=0,
    pxx=0,
    pxy=0,
    pxz=0,
    pyi=0,
    pyx=0,
    pyy=0,
    pyz=0,
    pzi=0,
    pzx=0,
    pzy=0,
    pzz=0,
    num_samples: int = 1,
):
    remainder = (
        1
        - pix
        - piy
        - piz
        - pxi
        - pxx
        - pxy
        - pxz
        - pyi
        - pyx
        - pyy
        - pyz
        - pzi
        - pzx
        - pzy
        - pzz
    )
    probs = jnp.array(
        [
            remainder,  # 00,00
            pzi,  # 10,00
            pxi,  # 01,00
            pyi,  # 11,00
            piz,  # 00,10
            pzz,  # 10,10
            pxz,  # 01,10
            pyz,  # 11,10
            pix,  # 00,01
            pzx,  # 10,01
            pxx,  # 01,01
            pyx,  # 11,01
            piy,  # 00,11
            pzy,  # 10,11
            pxy,  # 01,11
            pyy,  # 11,11
        ]
    )
    logits = jnp.log(probs)
    samples = jax.random.categorical(key, logits, shape=(num_samples,))
    bits = ((samples[:, None] >> jnp.arange(4)) & 1).astype(jnp.uint8)
    return bits


def x_error(p, num_samples: int = 1):
    return


print(pauli_channel_2(pxi=0.2, piy=0.6, num_samples=10))


# depolarize_1(p)
# [1, 0] p/3
# [0, 1] p/3
# [1, 1] p/3
