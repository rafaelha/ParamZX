import numpy as np
from pyzx.simulation.gen import make_g_list, chars
from pyzx.simulation.compile import compile_circuit


n_params = 5
g_list = make_g_list(n_params)
num_graphs = len(g_list)


# %% COMPILATION ------------------

circuit = compile_circuit(g_list, n_params, chars)

print("Compiled circuit:")
print(f"  {circuit.num_graphs} graphs, {circuit.n_params} parameters")
print(f"  {len(circuit.ab_terms)} AB terms")
print(f"  {len(circuit.c_terms)} C terms")
print(f"  {len(circuit.d_terms)} D terms")

param_vals = np.array([1, 0, 1, 0, 0], dtype=np.uint8)

# %%  VECTORIZED EXECUTION ------------------
print("\nExecuting (vectorized)...")


phase_lut = np.exp(1j * np.pi * np.arange(8) / 4)

# ============================================================================
# TYPE A/B: Node and Half-Pi Terms
# ============================================================================
inp = np.array([1, 1] + param_vals.tolist())
ab_eval = circuit.ab_terms * inp

ttype = ab_eval[:, 0]
const = ab_eval[:, 1]

rowsum = np.sum(ab_eval[:, 2:], axis=1) % 2
phase_idx = ((4 * rowsum + const) % 8) * ttype // 4

term_vals_ab = phase_lut[phase_idx]

term_vals_ab[ttype == 4] += 1

summands_ab = np.ones(circuit.num_graphs, dtype=complex)
np.multiply.at(summands_ab, circuit.ab_graph_ids, term_vals_ab)


# ============================================================================
# TYPE C: Pi-Pair Terms
# ============================================================================
inp = np.array([1] + param_vals.tolist() + [1] + param_vals.tolist())
c_eval = circuit.c_terms * inp

rowsum_a = np.sum(c_eval[:, : circuit.n_params + 1], axis=1) % 2
rowsum_b = np.sum(c_eval[:, circuit.n_params + 1 :], axis=1) % 2

term_vals_c = 1 - 2 * (rowsum_a * rowsum_b)

summands_c = np.ones(circuit.num_graphs, dtype=complex)
np.multiply.at(summands_c, circuit.c_graph_ids, term_vals_c)


# ============================================================================
# TYPE D: Phase Pairs
# ============================================================================
inp = np.array([1, 1] + (param_vals.tolist() * 2))
d_eval = circuit.d_terms * inp

n_ancil = 2

rowsum_a = np.sum(d_eval[:, n_ancil : circuit.n_params + n_ancil], axis=1) % 2
rowsum_b = np.sum(d_eval[:, circuit.n_params + n_ancil :], axis=1) % 2

alpha = (d_eval[:, 0] + rowsum_a * 4) % 8
beta = (d_eval[:, 1] + rowsum_b * 4) % 8
gamma = (alpha + beta) % 8

# Vectorized complex exponentials
term_vals_d = 1.0 + phase_lut[alpha] + phase_lut[beta] - phase_lut[gamma]

# Aggregate by graph_id
summands_d = np.ones(circuit.num_graphs, dtype=complex)
np.multiply.at(summands_d, circuit.d_graph_ids, term_vals_d)


# ============================================================================
# FINAL RESULT
# ============================================================================

# Vectorized final computation
root2 = np.sqrt(2)
contributions = (
    summands_ab
    * summands_c
    * summands_d
    * phase_lut[circuit.phase_indices]
    * root2**circuit.power2
    * circuit.floatfactor
)

result = np.sum(contributions)

print(f"Result: {result}")
assert np.isclose(result, 0.0683018803681063)
print("✓ Vectorized evaluation matches expected result!")
