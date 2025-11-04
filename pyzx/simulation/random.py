import jax
import jax.numpy as jnp
import abc


class Channel(abc.ABC):
    logits: jnp.ndarray

    @abc.abstractmethod
    def sample(self, num_samples: int = 1):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(probs={jnp.exp(self.logits)})"


class PauliChannel1:
    def __init__(self, px: float, py: float, pz: float, key: jax.random.PRNGKey):
        self._key = key
        probs = jnp.array([1 - px - py - pz, pz, px, py])
        self.logits = jnp.log(probs)

    def sample(self, num_samples: int = 1):
        self._key, subkey = jax.random.split(self._key)
        samples = jax.random.categorical(subkey, self.logits, shape=(num_samples,))
        bits = ((samples[:, None] >> jnp.arange(2)) & 1).astype(jnp.uint8)
        return bits


class Depolarize1(PauliChannel1):
    def __init__(self, p: float, key: jax.random.PRNGKey):
        super().__init__(p / 3, p / 3, p / 3, key=key)


class PauliChannel2:
    def __init__(
        self,
        pix: float,
        piy: float,
        piz: float,
        pxi: float,
        pxx: float,
        pxy: float,
        pxz: float,
        pyi: float,
        pyx: float,
        pyy: float,
        pyz: float,
        pzi: float,
        pzx: float,
        pzy: float,
        pzz: float,
        key: jax.random.PRNGKey,
    ):
        self._key = key
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
        self.logits = jnp.log(probs)

    def sample(self, num_samples: int = 1):
        self._key, subkey = jax.random.split(self._key)
        samples = jax.random.categorical(subkey, self.logits, shape=(num_samples,))
        bits = ((samples[:, None] >> jnp.arange(4)) & 1).astype(jnp.uint8)
        return bits


class Depolarize2(PauliChannel2):
    def __init__(self, p: float, key: jax.random.PRNGKey):
        super().__init__(
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            p / 15,
            key=key,
        )


class Error(Channel):
    def __init__(self, p: float, key: jax.random.PRNGKey):
        self._key = key
        self.p = p

    def sample(self, num_samples: int = 1, num_bits: int = 1):
        self._key, subkey = jax.random.split(self._key)
        samples = jax.random.bernoulli(subkey, self.p, shape=(num_samples,)).astype(
            jnp.uint8
        )
        return jnp.repeat(samples[:, None], num_bits, axis=1)


key = jax.random.key(0)
error = Error(0.1, key)
print(error.sample(10, 3))
