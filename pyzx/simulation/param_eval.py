# %%
import numpy as np
from pyzx.simulation.gen import make_g_list, chars


# %%
n_params = 5
g_list = make_g_list(n_params)
num_graphs = len(g_list)


print("Number of terms per graph...")
print("(1+e^x)/e^((1/2)x)\t (e^(x*y))\t phPair")
# node (1+e^x) - scalar.phasenodes
# B    e^x/2   - scalar.phasevars_halfpi[1] [3]
# C    e^(x*y) - scalar.phasevars_pi_pair
# D    (1+e^x+e^y+e^(x+y)) - scalar.phasepairs

data = []
for i in range(num_graphs):
    g_i = g_list[i]
    count_node = len(g_i.scalar.phasenodes)
    count_half_pi = len(g_i.scalar.phasevars_halfpi[1]) + len(
        g_i.scalar.phasevars_halfpi[3]
    )
    count_pi_pair = len(g_i.scalar.phasevars_pi) + len(g_i.scalar.phasevars_pi_pair)
    count_phase_pair = len(g_i.scalar.phasepairs)

    row = [count_node + count_half_pi, count_pi_pair, count_phase_pair]
    data.append(row)
data = np.array(data)
print(np.max(data, axis=0))


# %% COMPILATION ------------------

# node and half_pi terms
# graph_id, multiplier, const, bitstr
compiled_ab = []

char_to_idx = {char: i for i, char in enumerate(chars)}

for i in range(num_graphs):
    g_i = g_list[i]
    for term in range(len(g_i.scalar.phasenodevars)):
        bitstr = [0] * n_params
        for v in g_i.scalar.phasenodevars[term]:
            bitstr[char_to_idx[v]] = 1
        const_term = int(g_i.scalar.phasenodes[term] * 4)

        row_data = []
        row_data.append(i)  # Graph_id
        row_data.append(4)  # type: node type = 4 (as this is a multiplier, 4/4 = 1x)
        row_data.append(const_term)  # const
        row_data.extend(bitstr)
        compiled_ab.append(row_data)

    for j in [1, 3]:
        for term in range(len(g_i.scalar.phasevars_halfpi[j])):
            bitstr = [0] * n_params
            for v in g_i.scalar.phasevars_halfpi[j][term]:
                bitstr[char_to_idx[v]] = 1
            const_term = 0
            # (1/2)*4 = 2 or (3/2)*4 = 6 - this (divided by 4) is a multiplier
            ttype = int((j / 2) * 4)

            row_data = []
            row_data.append(i)  # Graph_id
            row_data.append(ttype)  # type
            row_data.append(const_term)  # const
            row_data.extend(bitstr)
            compiled_ab.append(row_data)


compiled_ab = np.array(compiled_ab)

char_to_idx = {char: i + 1 for i, char in enumerate(chars)}
char_to_idx["1"] = 0
compiled_c = []

for i in range(num_graphs):
    graph = g_list[i].copy()

    for p_set in graph.scalar.phasevars_pi_pair:
        bitstr = [0] * (n_params + 1) * 2
        for p in p_set[0]:
            bitstr[char_to_idx[p]] = 1
        for p in p_set[1]:
            bitstr[char_to_idx[p] + (n_params + 1)] = 1

        row_data = []
        row_data.append(i)  # Graph_id
        row_data.extend(bitstr)
        compiled_c.append(row_data)

compiled_c = np.array(compiled_c)

n_ancil = 3  # Extra bits needed: Multiplier, const term alpha, const term beta
compiled_d = []
char_to_idx = {char: i for i, char in enumerate(chars)}

for i in range(num_graphs):
    graph = g_list[i]
    for pp in range(len(graph.scalar.phasepairs)):
        bitstr = [0] * n_params * 2
        for v in graph.scalar.phasepairs[pp].paramsA:
            bitstr[char_to_idx[v]] = 1
        for v in graph.scalar.phasepairs[pp].paramsB:
            bitstr[char_to_idx[v] + n_params] = 1
        const_term_a = int(graph.scalar.phasepairs[pp].alpha)
        const_term_b = int(graph.scalar.phasepairs[pp].beta)

        row_data = []
        row_data.append(i)  # Graph_id
        row_data.append(const_term_a)  # const term alpha
        row_data.append(const_term_b)  # const term beta
        row_data.extend(bitstr)
        compiled_d.append(row_data)

compiled_d = np.array(compiled_d)


compiled_static = []

for i in range(num_graphs):
    graph = g_list[i]
    row_data = []
    row_data.append(int(graph.scalar.phase * 4))  # phase
    row_data.append(graph.scalar.power2)  # power2
    row_data.append(graph.scalar.floatfactor.real)  # float factor (real part)
    row_data.append(graph.scalar.floatfactor.imag)  # float factor (imag part)
    compiled_static.append(row_data)

compiled_static = np.array(compiled_static)

# %%  VECTORIZED EXECUTION ------------------
print("Executing (vectorized)...")

# Parameter values
param_vals = np.array([1, 0, 1, 0, 0])

# Pre-compute phase lookup table (e^(i*pi*k/4) for k=0..7)
phase_lut = np.exp(1j * np.pi * np.arange(8) / 4)

# ============================================================================
# TYPE A/B: Node and Half-Pi Terms
# ============================================================================
inp = np.array([1, 1, 1] + param_vals.tolist())
compiled_ab_eval = compiled_ab * inp

# Extract columns
graph_ids_ab = compiled_ab_eval[:, 0]
ttype = compiled_ab_eval[:, 1]
const = compiled_ab_eval[:, 2]

# Vectorized phase calculation
rowsum = np.sum(compiled_ab_eval[:, 3:], axis=1) % 2
phase_idx = ((4 * rowsum + const) % 8) * ttype // 4

# Lookup exponentials
term_vals_ab = phase_lut[phase_idx]

# Handle special case (TODO from original)
term_vals_ab[ttype == 4] += 1

# Aggregate by graph_id using product
summands_ab = np.ones(num_graphs, dtype=complex)
np.multiply.at(summands_ab, graph_ids_ab, term_vals_ab)


# ============================================================================
# TYPE C: Pi-Pair Terms
# ============================================================================
inp = np.array([1, 1] + param_vals.tolist() + [1] + param_vals.tolist())
compiled_c_eval = compiled_c * inp

graph_ids_c = compiled_c_eval[:, 0]
rowsum_a = np.sum(compiled_c_eval[:, 1 : n_params + 2], axis=1) % 2
rowsum_b = np.sum(compiled_c_eval[:, n_params + 2 :], axis=1) % 2

# XOR logic: (-1)^(A AND B) = 1 - 2*(A AND B)
term_vals_c = 1 - 2 * (rowsum_a * rowsum_b)

# Aggregate by graph_id
summands_c = np.ones(num_graphs, dtype=complex)
np.multiply.at(summands_c, graph_ids_c, term_vals_c)


# ============================================================================
# TYPE D: Phase Pairs
# ============================================================================
inp = np.array([1, 1, 1] + (param_vals.tolist() * 2))
compiled_d_eval = compiled_d * inp

graph_ids_d = compiled_d_eval[:, 0]
n_ancil = 3

rowsum_a = np.sum(compiled_d_eval[:, n_ancil : n_params + n_ancil], axis=1) % 2
rowsum_b = np.sum(compiled_d_eval[:, n_params + n_ancil :], axis=1) % 2

alpha = (compiled_d_eval[:, 1] + rowsum_a * 4) % 8
beta = (compiled_d_eval[:, 2] + rowsum_b * 4) % 8
gamma = (alpha + beta) % 8

# Vectorized complex exponentials
term_vals_d = 1.0 + phase_lut[alpha] + phase_lut[beta] - phase_lut[gamma]

# Aggregate by graph_id
summands_d = np.ones(num_graphs, dtype=complex)
np.multiply.at(summands_d, graph_ids_d, term_vals_d)


# ============================================================================
# FINAL RESULT
# ============================================================================
# Extract static terms into arrays
phases = compiled_static[:, 0].astype(int)
power2s = compiled_static[:, 1].astype(int)
floatfactors = compiled_static[:, 2] + 1j * compiled_static[:, 3]

# Vectorized final computation
root2 = np.sqrt(2)
contributions = (
    summands_ab
    * summands_c
    * summands_d
    * phase_lut[phases]
    * root2**power2s
    * floatfactors
)

result = np.sum(contributions)

print(f"Result: {result}")
assert np.isclose(result,0.0683018803681063)
print("✓ Vectorized evaluation matches expected result!")
