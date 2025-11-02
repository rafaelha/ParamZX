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


# %%  VECTORIZED EXECUTION ------------------
print("\nExecuting (vectorized)...")

# Parameter values
param_vals = np.array([1, 0, 1, 0, 0])

# Pre-compute phase lookup table (e^(i*pi*k/4) for k=0..7)
phase_lut = np.exp(1j * np.pi * np.arange(8) / 4)

# ============================================================================
# TYPE A/B: Node and Half-Pi Terms
# ============================================================================
if len(circuit.ab_terms) > 0:
    inp = np.array([1, 1] + param_vals.tolist())
    ab_eval = circuit.ab_terms * inp

    # Extract columns
    ttype = ab_eval[:, 0]
    const = ab_eval[:, 1]

    # Vectorized phase calculation
    rowsum = np.sum(ab_eval[:, 2:], axis=1) % 2
    phase_idx = ((4 * rowsum + const) % 8) * ttype // 4

    # Lookup exponentials
    term_vals_ab = phase_lut[phase_idx]

    # Handle special case (TODO from original)
    term_vals_ab[ttype == 4] += 1

    # Aggregate by graph_id using product
    summands_ab = np.ones(circuit.num_graphs, dtype=complex)
    np.multiply.at(summands_ab, circuit.ab_graph_ids, term_vals_ab)
else:
    summands_ab = np.ones(circuit.num_graphs, dtype=complex)


# ============================================================================
# TYPE C: Pi-Pair Terms
# ============================================================================
if len(circuit.c_terms) > 0:
    inp = np.array([1] + param_vals.tolist() + [1] + param_vals.tolist())
    c_eval = circuit.c_terms * inp

    rowsum_a = np.sum(c_eval[:, : circuit.n_params + 1], axis=1) % 2
    rowsum_b = np.sum(c_eval[:, circuit.n_params + 1 :], axis=1) % 2

    # XOR logic: (-1)^(A AND B) = 1 - 2*(A AND B)
    term_vals_c = 1 - 2 * (rowsum_a * rowsum_b)

    # Aggregate by graph_id
    summands_c = np.ones(circuit.num_graphs, dtype=complex)
    np.multiply.at(summands_c, circuit.c_graph_ids, term_vals_c)
else:
    summands_c = np.ones(circuit.num_graphs, dtype=complex)


# ============================================================================
# TYPE D: Phase Pairs
# ============================================================================
if len(circuit.d_terms) > 0:
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
else:
    summands_d = np.ones(circuit.num_graphs, dtype=complex)


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
