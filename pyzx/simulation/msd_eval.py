from pyzx.simulation.msd import msd_graph
import numpy as np
from pyzx.simulation.gen import make_parametrized, get_g_list
from pyzx.simulation.compile import compile_circuit
from pyzx.simulation.evaluate import evaluate_batch, evaluate
from pyzx.simulation.tictoc import tic, toc
import jax
import jax.numpy as jnp
import pyzx as zx

g = msd_graph(17)
n_qubits = g.num_outputs()
zx.draw(g)

tic()
compiled_circuits = []
for i in range(1, n_qubits + 1):
    g_param = make_parametrized(g, i)
    g_list = get_g_list(g_param)
    circuit = compile_circuit(g_list, i)
    compiled_circuits.append(circuit)
toc(f"Compiled {n_qubits} circuits in")

c_graphs = [c.num_graphs for c in compiled_circuits]
c_params = [c.n_params for c in compiled_circuits]
print(f"Graphs: {np.mean(c_graphs):.1f} ({np.min(c_graphs)}/{np.max(c_graphs)})")
print(f"Params: {np.mean(c_params):.1f} ({np.min(c_params)}/{np.max(c_params)})")


# %%
batch_size = 5
s = jnp.zeros((batch_size, 0), dtype=jnp.uint8)
zeros = jnp.zeros((batch_size, 1), dtype=jnp.uint8)
p_prev = jnp.ones((batch_size,), dtype=jnp.float32)
tic()

key = jax.random.key(0)
for i in range(n_qubits):
    state = jnp.hstack([s, zeros])

    p_batch = evaluate_batch(compiled_circuits[i], state)

    p0 = jnp.abs(p_batch / p_prev)
    m = jax.random.bernoulli(key, p=1 - p0).astype(jnp.uint8)
    s = jnp.hstack([s, m[:, None]])
    p_prev = jnp.where(m == 0, p_prev * p0, p_prev * (1 - p0))

toc()
# %%
# print each row of s as a bitstring
for row in s:
    print("".join(map(str, row)))
