import pyzx as zx
from pyzx import VertexType, EdgeType
from functools import wraps
from typing import Iterable
import jax
from pyzx.simulation.random import (
    PauliChannel1,
    PauliChannel2,
    Depolarize1,
    Depolarize2,
    Error,
)


g = zx.Graph()
g.add_vertex(zx.VertexType.BOUNDARY, qubit=0, row=0)


def accepts_qubit_list(func):
    """Decorator that allows a method to accept either a single qubit or a list of qubits.

    When a list is provided, the decorated method is called once for each qubit in the list.

    Example:
        @accepts_qubit_list
        def h(self, qubit: int):
            # implementation

        dem.h(0)        # Applies to single qubit
        dem.h([0,1,2])  # Applies to qubits 0, 1, and 2
    """

    @wraps(func)
    def wrapper(self, qubit, *args, **kwargs):
        # Check if qubit is iterable (list, tuple, etc.) but not a string
        if isinstance(qubit, Iterable):
            # Apply the function to each qubit in the list
            for q in qubit:
                func(self, q, *args, **kwargs)
        else:
            # Single qubit case - call function normally
            func(self, qubit, *args, **kwargs)

    return wrapper


class Circ:

    def __init__(self, key: jax.random.PRNGKey):
        self.key = key
        self.g = zx.Graph()
        self.last_vertex: dict[int, int] = {}
        self.last_row: dict[int, int] = {}
        self.errors = []
        self.num_error_bits = 0

    def _last_row(self, qubit: int):
        return self.g.row(self.last_vertex[qubit])

    def _last_edge(self, qubit: int):
        edges = self.g.incident_edges(self.last_vertex[qubit])
        assert len(edges) == 1
        return edges[0]

    def _add_dummy(self, qubit: int, row: int | None = None):
        if row is None:
            row = self._last_row(qubit) + 1
        v1 = self.g.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=row)
        self.last_vertex[qubit] = v1
        return v1

    def _initialize_qubit(self, qubit: int):
        v1 = self.g.add_vertex(VertexType.X, qubit=qubit, row=0)
        v2 = self.g.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=1)
        self.g.add_edge((v1, v2))
        self.last_vertex[qubit] = v2

    @accepts_qubit_list
    def h(self, qubit: int):
        g = self.g
        if qubit not in self.last_vertex:
            self._initialize_qubit(qubit)

        e = self._last_edge(qubit)
        g.set_edge_type(e, EdgeType.HADAMARD)

    def x(self, qubit: int):
        g = self.g
        if qubit not in self.last_vertex:
            self._initialize_qubit(qubit)
        v1 = self.last_vertex[qubit]
        g.set_type(v1, VertexType.X)
        g.set_phase(v1, 1)
        v2 = self._add_dummy(qubit)
        g.add_edge((v1, v2))

    @accepts_qubit_list
    def reset(self, qubit: int):
        g = self.g
        if qubit not in self.last_vertex:
            self._initialize_qubit(qubit)
            g.set_type(self.last_vertex[qubit], VertexType.BOUNDARY)
        else:
            v1 = self.last_vertex[qubit]
            g.set_type(v1, VertexType.Z)
            r = self._last_row(qubit)
            v2 = g.add_vertex(VertexType.X, qubit=qubit, row=r + 1)
            v3 = self._add_dummy(qubit, r + 2)

            g.add_edge((v2, v3))

    @accepts_qubit_list
    def mr(self, qubit: int):
        g = self.g
        if qubit not in self.last_vertex:
            self._initialize_qubit(qubit)
            g.set_type(self.last_vertex[qubit], VertexType.BOUNDARY)
        v1 = self.last_vertex[qubit]
        g.set_type(v1, VertexType.Z)
        r = self._last_row(qubit)
        v2 = g.add_vertex(VertexType.X, qubit=qubit, row=r + 1)
        v3 = self._add_dummy(qubit, r + 2)

        g.add_edge((v2, v3))

    @accepts_qubit_list
    def m(self, qubit: int):
        g = self.g
        if qubit not in self.last_vertex:
            self._initialize_qubit(qubit)
        v1 = self.last_vertex[qubit]
        g.set_type(v1, VertexType.Z)
        v2 = self._add_dummy(qubit)
        g.add_edge((v1, v2))

    def cnot(self, control: int, target: int):
        g = self.g
        if control not in self.last_vertex:
            self._initialize_qubit(control)
        if target not in self.last_vertex:
            self._initialize_qubit(target)

        lr1 = self._last_row(control)
        lr2 = self._last_row(target)
        r = max(lr1, lr2)

        v1 = self.last_vertex[control]
        v2 = self.last_vertex[target]
        g.set_type(v1, VertexType.Z)
        g.set_type(v2, VertexType.X)
        g.set_row(v1, r)
        g.set_row(v2, r)
        g.add_edge((v1, v2))

        v3 = self._add_dummy(control, r + 1)
        v4 = self._add_dummy(target, r + 1)
        g.add_edge((v1, v3))
        g.add_edge((v2, v4))

    def _error(self, qubit: int, error_type: VertexType, phase: str):
        if qubit not in self.last_vertex:
            self._initialize_qubit(qubit)
        g = self.g
        v1 = self.last_vertex[qubit]
        v2 = self._add_dummy(qubit)
        g.add_edge((v1, v2))

        g.set_type(v1, error_type)
        g.set_phase(v1, phase)

    @accepts_qubit_list
    def x_error(self, qubit: int, p: float):
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(Error(p, subkey))
        self._error(qubit, VertexType.X, f"e{self.num_error_bits}")
        self.num_error_bits += 1

    @accepts_qubit_list
    def z_error(self, qubit: int, p: float):
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(Error(p, subkey))
        self._error(qubit, VertexType.Z, f"e{self.num_error_bits}")
        self.num_error_bits += 1

    @accepts_qubit_list
    def y_error(self, qubit: int, p: float):
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(Error(p, subkey))
        self._error(qubit, VertexType.Z, f"e{self.num_error_bits}")
        self._error(qubit, VertexType.X, f"e{self.num_error_bits}")
        self.num_error_bits += 1

    @accepts_qubit_list
    def depolarize1(self, qubit: int, p: float):
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(Depolarize1(p, subkey))
        self._error(qubit, VertexType.Z, f"e{self.num_error_bits}")
        self._error(qubit, VertexType.X, f"e{self.num_error_bits + 1}")
        self.num_error_bits += 2

    def depolarize2(self, qubit_i: int, qubit_j: int, p: float):
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(Depolarize2(p, subkey))
        self._error(qubit_i, VertexType.Z, f"e{self.num_error_bits}")
        self._error(qubit_i, VertexType.X, f"e{self.num_error_bits + 1}")
        self._error(qubit_j, VertexType.Z, f"e{self.num_error_bits + 2}")
        self._error(qubit_j, VertexType.X, f"e{self.num_error_bits + 3}")
        self.num_error_bits += 4

    def identity(self, qubit):
        v = self.last_vertex[qubit]
        self.g.set_row(v, self._last_row(qubit) + 1)

    def tick(self):
        r = max(self._last_row(q) for q in self.last_vertex)
        for q in self.last_vertex:
            self.g.set_row(self.last_vertex[q], r)

    def diagram(self, labels=False):
        g = self.g.copy()
        for q in self.last_vertex:
            g.set_type(self.last_vertex[q], VertexType.Z)
        zx.draw(g, labels=labels)


key = jax.random.key(0)
rep_code = Circ(key)

rep_code.reset(range(5))
rep_code.x_error(range(1, 4), 0.1)
rep_code.x(0)
# rep_code.z_error(range(1, 4))
rep_code.tick()

for r in range(1):
    rep_code.cnot(1, 0)
    rep_code.cnot(2, 4)
    rep_code.cnot(2, 0)
    rep_code.cnot(3, 4)
    rep_code.depolarize2(3, 4, 0.1)

    rep_code.tick()
    rep_code.x_error([0, 4], 0.1)
    rep_code.tick()
    # rep_code.x_error([1, 2, 3])

    rep_code.tick()
    rep_code.mr([0, 4])
    rep_code.tick()

rep_code.m([1, 2, 3])

rep_code.diagram(labels=False)
