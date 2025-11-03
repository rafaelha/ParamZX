from fractions import Fraction
import random
import time
import pyzx as zx
from pyzx.graph.base import BaseGraph
import numpy as np


def add_sqrt_y_dag(circ: zx.Circuit, q: int | np.ndarray):
    if not isinstance(q, np.ndarray):
        q = np.array([q])
    for i in q:
        circ.add_gate("Z", i)
        circ.add_gate("YPhase", i, phase=Fraction(-1, 2))
        circ.add_gate("Z", i)
    return circ


def add_sqrt_y(circ: zx.Circuit, q: int | np.ndarray):
    if not isinstance(q, np.ndarray):
        q = np.array([q])
    for i in q:
        circ.add_gate("Z", i)
        circ.add_gate("YPhase", i, phase=Fraction(1, 2))
        circ.add_gate("Z", i)
    return circ


def add_sqrt_x(circ: zx.Circuit, q: int | np.ndarray):
    if not isinstance(q, np.ndarray):
        q = np.array([q])
    for i in q:
        circ.add_gate("XPhase", i, phase=Fraction(1, 2))
    return circ


def add_sqrt_x_dag(circ: zx.Circuit, q: int | np.ndarray):
    if not isinstance(q, np.ndarray):
        q = np.array([q])
    for i in q:
        circ.add_gate("XPhase", i, phase=Fraction(-1, 2))
    return circ


def add_cz(circ: zx.Circuit, q1: np.ndarray, q2: np.ndarray):
    for i, j in zip(q1, q2):
        circ.add_gate("CZ", i, j)
    return circ


def encode(circ: zx.Circuit, qs: np.ndarray):
    """
    Encode the logical state into the physical qubits using the distance 3 or 5 color code.

    Args:
        circ: The ZX circuit to add the encoding to.
        qs: The physical qubits to encode the logical state into. Must be either of
        length 7 or 17.
    """
    if len(qs) == 7:
        for q in qs[:-1]:
            add_sqrt_y_dag(circ, q)

        circ.add_gate("CZ", qs[1], qs[2])
        circ.add_gate("CZ", qs[3], qs[4])
        circ.add_gate("CZ", qs[5], qs[6])

        add_sqrt_y(circ, qs[6])

        circ.add_gate("CZ", qs[0], qs[3])
        circ.add_gate("CZ", qs[2], qs[5])
        circ.add_gate("CZ", qs[4], qs[6])

        for q in qs[2:]:
            add_sqrt_y(circ, q)

        circ.add_gate("CZ", qs[0], qs[1])
        circ.add_gate("CZ", qs[2], qs[3])
        circ.add_gate("CZ", qs[4], qs[5])

        for q in qs[[1, 2, 4]]:
            add_sqrt_y(circ, q)

    elif len(qs) == 17:
        for q in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            add_sqrt_y(circ, q)
        for i, j in ([1, 3], [7, 10], [12, 14], [13, 16]):
            circ.add_gate("CZ", qs[i], qs[j])
        for i in [7, 16]:
            add_sqrt_y_dag(circ, qs[i])
        for i, j in ([4, 7], [8, 10], [11, 14], [15, 16]):
            circ.add_gate("CZ", qs[i], qs[j])
        for i in [4, 10, 14, 16]:
            add_sqrt_y_dag(circ, qs[i])
        for i, j in ([2, 4], [6, 8], [7, 9], [10, 13], [14, 16]):
            circ.add_gate("CZ", qs[i], qs[j])
        for i in [3, 6, 9, 10, 12, 13]:
            add_sqrt_y(circ, qs[i])
        for i, j in ([0, 2], [3, 6], [5, 8], [10, 12], [11, 13]):
            circ.add_gate("CZ", qs[i], qs[j])
        for i in [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 14]:
            add_sqrt_y(circ, qs[i])
        for i, j in ([0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [12, 15]):
            circ.add_gate("CZ", qs[i], qs[j])
        for i in [0, 2, 5, 6, 8, 10, 12]:
            add_sqrt_y_dag(circ, qs[i])

    else:
        raise ValueError(f"Unsupported number of qubits: {len(qs)}")

    return circ


def msd_circuit(qubits_per_code_block: int = 7) -> zx.Circuit:
    """
    Create the logical magic state distillation circuit of 5 logical code blocks, where
    each logical code block comprises `qubits_per_code_block` qubits (either 7 or 17).

    Args:
        qubits_per_code_block: The number of qubits per logical code block.

    Returns:
        A zx.Circuit object representing the logical magic state distillation circuit.
    """
    n = qubits_per_code_block
    qubits = np.arange(n * 5)
    # split qubits in 5 groups of n qubits
    ql = np.array_split(qubits, 5)

    circ = zx.Circuit(len(qubits))
    for q in ql:
        # Prepare a logical T state
        encoding_qubit = q[6] if len(q) == 7 else q[7]
        circ.add_gate("H", encoding_qubit)
        circ.add_gate("T", encoding_qubit)
        encode(circ, q)

    for i in [0, 1, 4]:
        add_sqrt_x(circ, ql[i])

    add_cz(circ, ql[0], ql[1])
    add_cz(circ, ql[2], ql[3])

    add_sqrt_y(circ, ql[0])
    add_sqrt_y(circ, ql[3])

    add_cz(circ, ql[0], ql[2])
    add_cz(circ, ql[3], ql[4])

    add_sqrt_x_dag(circ, ql[0])

    add_cz(circ, ql[0], ql[4])
    add_cz(circ, ql[1], ql[3])

    for i in range(5):
        add_sqrt_x_dag(circ, ql[i])

    return circ


def msd_graph(qubits_per_code_block: int = 7) -> BaseGraph:
    """
    Returns a ZX graph represention the statevector of the logical magic state
    distillation circuit.

    Args:
        qubits_per_code_block: The number of qubits per logical code block.

    Returns:
        A ZX graph representing the statevector of the logical magic state distillation circuit.
    """
    c = msd_circuit(qubits_per_code_block)
    g = c.to_graph()
    g.apply_state("0" * qubits_per_code_block * 5)
    return g


def stab_rank_dfs(g: BaseGraph):
    """
    Perform a stabilizer rank decomposition of the ZX graph. This will return the same
    result as `g.to_tensor()` but scales exponentially in the number of T gates and
    only polynomial in the graph's vertices.

    Args:
        g: A ZX graph that represents a scalar (i.e. no inputs and outputs)

    Returns:
        The scalar value of the ZX graph.
    """
    if zx.tcount(g) > 0:
        gsum = zx.simulate.replace_magic_states(g)
        vals = 0
        for graph in gsum.graphs:
            zx.full_reduce(graph)
            vals += stab_rank_dfs(graph)
        return vals
    else:
        # graph should not have any vertices, just extract the scalar value
        return g.to_tensor()


def sample(g: BaseGraph) -> str:
    """
    Given a ZX graph that represents a statevector (i.e. no inputs), sample a bitstring
    from its probability distribution.

    Args:
        g: A ZX graph that represents a statevector (i.e. no inputs).

    Returns:
        A bitstring sampled from the probability distribution of the statevector.
    """
    n_qubits = g.num_outputs()
    s = ""
    p_prev = 1

    for _ in range(n_qubits):
        # Build the ZX diagram for the marginal probability P(x_1 ... x_k)
        g_ = g.copy()
        g_.apply_effect(s + "0" + "/" * (n_qubits - len(s) - 1))
        g_ += g_.adjoint()
        zx.full_reduce(g_)

        p = stab_rank_dfs(g_)

        # Compute the conditional probability of measuring 0 from marginal probabilities
        p0 = abs(p / p_prev)
        # Sample the next bit and update the marginal probability
        m = random.choices([0, 1], weights=[p0, 1 - p0])[0]
        s += str(m)
        p_prev *= p0 if m == 0 else 1 - p0

    return s


if __name__ == "__main__":
    n = 17
    g = msd_graph(n)

    start = time.perf_counter()
    s = sample(g)
    duration = time.perf_counter() - start
    print(s)
    print(f"generated sample in: {duration:.2f} seconds")