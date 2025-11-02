from typing import NamedTuple
import numpy as np


class CompiledCircuit(NamedTuple):
    """JAX-compatible compiled circuit representation.

    All fields are static-shaped NumPy arrays, making them directly
    convertible to JAX arrays for GPU execution and JIT compilation.
    """

    # Metadata
    num_graphs: int
    n_params: int

    # Type A/B: Node and Half-Pi Terms
    # Shape: (n_ab_terms, 2 + n_params)
    # Columns: [term_type, const_phase, param_bit_0, ..., param_bit_{n_params-1}]
    ab_terms: np.ndarray  # dtype: uint8
    ab_graph_ids: np.ndarray  # dtype: int32, shape: (n_ab_terms,)

    # Type C: Pi-Pair Terms
    # Shape: (n_c_terms, 2*(n_params+1))
    # Columns: [param_bits_a[0..n_params], param_bits_b[0..n_params]]
    c_terms: np.ndarray  # dtype: uint8
    c_graph_ids: np.ndarray  # dtype: int32, shape: (n_c_terms,)

    # Type D: Phase Pairs
    # Shape: (n_d_terms, 2 + 2*n_params)
    # Columns: [const_alpha, const_beta, param_bits_a[0..n_params-1], param_bits_b[0..n_params-1]]
    d_terms: np.ndarray  # dtype: uint8
    d_graph_ids: np.ndarray  # dtype: int32, shape: (n_d_terms,)

    # Static per-graph data
    # Shape: (num_graphs,)
    phase_indices: np.ndarray  # dtype: uint8 (values 0-7)
    power2: np.ndarray  # dtype: int8
    floatfactor: np.ndarray  # dtype: complex128


def compile_circuit(g_list, n_params, chars):
    """Compile graph list into JAX-compatible structure."""
    num_graphs = len(g_list)
    char_to_idx = {char: i for i, char in enumerate(chars)}

    # ========================================================================
    # Type A/B compilation
    # ========================================================================
    compiled_ab = []
    g_coord_ab = []

    for i in range(num_graphs):
        g_i = g_list[i]
        for term in range(len(g_i.scalar.phasenodevars)):
            bitstr = [0] * n_params
            for v in g_i.scalar.phasenodevars[term]:
                bitstr[char_to_idx[v]] = 1
            const_term = int(g_i.scalar.phasenodes[term] * 4)

            g_coord_ab.append(i)
            row_data = [4, const_term] + bitstr
            compiled_ab.append(row_data)

        for j in [1, 3]:
            for term in range(len(g_i.scalar.phasevars_halfpi[j])):
                bitstr = [0] * n_params
                for v in g_i.scalar.phasevars_halfpi[j][term]:
                    bitstr[char_to_idx[v]] = 1
                const_term = 0
                ttype = int((j / 2) * 4)

                g_coord_ab.append(i)
                row_data = [ttype, const_term] + bitstr
                compiled_ab.append(row_data)

    ab_terms = (
        np.array(compiled_ab, dtype=np.uint8)
        if compiled_ab
        else np.zeros((0, 2 + n_params), dtype=np.uint8)
    )
    ab_graph_ids = np.array(g_coord_ab, dtype=np.int32)

    # ========================================================================
    # Type C compilation
    # ========================================================================
    char_to_idx_c = {char: i + 1 for i, char in enumerate(chars)}
    char_to_idx_c["1"] = 0
    compiled_c = []
    g_coord_c = []

    for i in range(num_graphs):
        graph = g_list[i]
        for p_set in graph.scalar.phasevars_pi_pair:
            bitstr = [0] * (n_params + 1) * 2
            for p in p_set[0]:
                bitstr[char_to_idx_c[p]] = 1
            for p in p_set[1]:
                bitstr[char_to_idx_c[p] + (n_params + 1)] = 1

            g_coord_c.append(i)
            compiled_c.append(bitstr)

    c_terms = (
        np.array(compiled_c, dtype=np.uint8)
        if compiled_c
        else np.zeros((0, 2 * (n_params + 1)), dtype=np.uint8)
    )
    c_graph_ids = np.array(g_coord_c, dtype=np.int32)

    # ========================================================================
    # Type D compilation
    # ========================================================================
    compiled_d = []
    g_coord_d = []

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

            g_coord_d.append(i)
            row_data = [const_term_a, const_term_b] + bitstr
            compiled_d.append(row_data)

    d_terms = (
        np.array(compiled_d, dtype=np.uint8)
        if compiled_d
        else np.zeros((0, 2 + 2 * n_params), dtype=np.uint8)
    )
    d_graph_ids = np.array(g_coord_d, dtype=np.int32)

    # ========================================================================
    # Static data
    # ========================================================================
    phase_indices = np.array([int(g.scalar.phase * 4) for g in g_list], dtype=np.uint8)
    power2 = np.array([g.scalar.power2 for g in g_list], dtype=np.int8)
    floatfactor = np.array([g.scalar.floatfactor for g in g_list], dtype=np.complex128)

    return CompiledCircuit(
        num_graphs=num_graphs,
        n_params=n_params,
        ab_terms=ab_terms,
        ab_graph_ids=ab_graph_ids,
        c_terms=c_terms,
        c_graph_ids=c_graph_ids,
        d_terms=d_terms,
        d_graph_ids=d_graph_ids,
        phase_indices=phase_indices,
        power2=power2,
        floatfactor=floatfactor,
    )
