import jax
import jax.numpy as jnp
from pyzx.simulation.gen import make_g_list, chars
from pyzx.simulation.compile import compile_circuit, CompiledCircuit

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

param_vals = jnp.array([1, 0, 1, 0, 0], dtype=jnp.uint8)


# %%  JAX EVALUATION FUNCTION ------------------


@jax.jit
def evaluate(circuit: CompiledCircuit, param_vals):
    """Evaluate the circuit with given parameter values."""
    n_params = len(param_vals)
    num_graphs = len(circuit.power2)

    # Pre-compute phase lookup table
    phase_lut = jnp.exp(1j * jnp.pi * jnp.arange(8) / 4)

    # ====================================================================
    # TYPE A/B: Node and Half-Pi Terms
    # ====================================================================
    inp = jnp.array([1, 1]).astype(jnp.uint8)
    inp = jnp.concatenate([inp, param_vals])
    ab_eval = circuit.ab_terms * inp

    ttype = ab_eval[:, 0]
    const = ab_eval[:, 1]

    rowsum = jnp.sum(ab_eval[:, 2:], axis=1) % 2
    phase_idx = ((4 * rowsum + const) % 8) * ttype // 4

    term_vals_ab = phase_lut[phase_idx]
    term_vals_ab = jnp.where(ttype == 4, term_vals_ab + 1, term_vals_ab)

    summands_ab = jax.ops.segment_prod(
        term_vals_ab,
        circuit.ab_graph_ids,
        num_segments=num_graphs,
        indices_are_sorted=True,
    )

    # ====================================================================
    # TYPE C: Pi-Pair Terms
    # ====================================================================
    inp = jnp.concatenate(
        [
            jnp.array([1], dtype=jnp.uint8),
            param_vals,
            jnp.array([1], dtype=jnp.uint8),
            param_vals,
        ]
    )
    c_eval = circuit.c_terms * inp

    rowsum_a = jnp.sum(c_eval[:, : n_params + 1], axis=1, dtype=jnp.uint8) % 2
    rowsum_b = jnp.sum(c_eval[:, n_params + 1 :], axis=1, dtype=jnp.uint8) % 2

    term_vals_c = 1 - 2 * (rowsum_a * rowsum_b).astype(jnp.complex64)

    summands_c = jax.ops.segment_prod(
        term_vals_c, circuit.c_graph_ids, num_segments=num_graphs
    )

    # ====================================================================
    # TYPE D: Phase Pairs
    # ====================================================================
    inp = jnp.concatenate([jnp.array([1, 1], dtype=jnp.uint8), param_vals, param_vals])
    d_eval = circuit.d_terms * inp

    n_ancil = 2
    rowsum_a = (
        jnp.sum(d_eval[:, n_ancil : n_params + n_ancil], axis=1, dtype=jnp.uint8) % 2
    )
    rowsum_b = jnp.sum(d_eval[:, n_params + n_ancil :], axis=1, dtype=jnp.uint8) % 2

    alpha = (d_eval[:, 0] + rowsum_a * 4) % 8
    beta = (d_eval[:, 1] + rowsum_b * 4) % 8
    gamma = (alpha + beta) % 8

    term_vals_d = 1.0 + phase_lut[alpha] + phase_lut[beta] - phase_lut[gamma]

    summands_d = jax.ops.segment_prod(
        term_vals_d, circuit.d_graph_ids, num_segments=num_graphs
    )

    # ====================================================================
    # FINAL RESULT
    # ====================================================================
    root2 = jnp.sqrt(2.0)
    contributions = (
        summands_ab
        * summands_c
        * summands_d
        * phase_lut[circuit.phase_indices]
        * root2**circuit.power2
        * circuit.floatfactor
    )
    print(jnp.max(jnp.abs(contributions.imag)))

    result = jnp.sum(contributions)
    return result


result = evaluate(circuit, param_vals)

print(f"Result: {result}")
assert jnp.isclose(result, 0.0683018803681063)
print("✓ JAX evaluation matches expected result!")
