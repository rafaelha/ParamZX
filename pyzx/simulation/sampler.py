from math import ceil
from pyzx.graph.base import BaseGraph
import numpy as np
from pyzx.simulation.gen import make_parametrized, get_g_list
from pyzx.simulation.compile import compile_circuit
from pyzx.simulation.evaluate import evaluate_batch
import jax
import jax.numpy as jnp


class Sampler:

    def __init__(self, g: BaseGraph):
        """Compile parametrized graphs"""
        self.compiled_circuits = []
        n_qubits = g.num_outputs()
        for i in range(1, n_qubits + 1):
            g_param = make_parametrized(g, i)
            g_list = get_g_list(g_param)
            circuit = compile_circuit(g_list, i)
            self.compiled_circuits.append(circuit)

        self._key = jax.random.key(0)

    def __repr__(self):
        c_graphs = [c.num_graphs for c in self.compiled_circuits]
        c_params = [c.n_params for c in self.compiled_circuits]
        num_circuits = len(self.compiled_circuits)
        return (
            f"CompiledSampler({num_circuits} qubits, {np.sum(c_graphs)} graphs, "
            f"{np.sum(c_params)} params)"
        )

    def sample_batch(self, batch_size: int) -> np.ndarray:
        s = jnp.zeros((batch_size, 0), dtype=jnp.uint8)
        zeros = jnp.zeros((batch_size, 1), dtype=jnp.uint8)
        p_prev = jnp.ones((batch_size,), dtype=jnp.float32)
        key = self._key

        for circuit in self.compiled_circuits:
            state = jnp.hstack([s, zeros])

            p_batch = evaluate_batch(circuit, state)

            p0 = jnp.abs(p_batch / p_prev)
            key, subkey = jax.random.split(key)
            m = jax.random.bernoulli(subkey, p=1 - p0).astype(jnp.uint8)
            s = jnp.hstack([s, m[:, None]])
            p_prev = jnp.where(m == 0, p_prev * p0, p_prev * (1 - p0))

        self._key = key
        return np.array(s)

    def sample(self, num_samples: int, batch_size: int = 100) -> np.ndarray:
        batches = []
        for _ in range(ceil(num_samples / batch_size)):
            batches.append(self.sample_batch(batch_size))
        return np.concatenate(batches)[:num_samples]
