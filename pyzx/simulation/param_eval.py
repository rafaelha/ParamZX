import jax.numpy as jnp
import time
import itertools
from pyzx.simulation.evaluate import evaluate_batch, evaluate
from pyzx.simulation.gen import make_g_list, chars
from pyzx.simulation.compile import compile_circuit

n_params = 5
g_list = make_g_list(n_params)
num_graphs = len(g_list)


circuit = compile_circuit(g_list, n_params, chars)

print("Compiled circuit:")
print(f"  {circuit.num_graphs} graphs, {circuit.n_params} parameters")
print(
    f"  {len(circuit.ab_term_types) + len(circuit.c_const_bits_a) + len(circuit.d_const_alpha)} terms"
)

param_vals = jnp.array([1, 0, 1, 0, 0], dtype=jnp.uint8)


n_params = circuit.n_params
bitstring_list = list(itertools.product([0, 1], repeat=n_params))
param_batch = jnp.array(bitstring_list, dtype=jnp.uint8)

result_batch = evaluate_batch(circuit, param_batch)

start_time = time.perf_counter()
result_batch = evaluate_batch(circuit, param_batch).block_until_ready()
end_time = time.perf_counter()
print(f"Time taken: {end_time - start_time:.2f} seconds")


batch_sum = jnp.sum(result_batch)
assert jnp.isclose(batch_sum, 1), f"Not normalized to 1. Instead got {batch_sum}"

result = evaluate(circuit, param_vals)

print(f"Result: {result}")
assert jnp.isclose(
    result, 0.0683018803681063
), f"Did not match 0.068. Instead got {result}"
print("✓ JAX evaluation matches expected result!")
