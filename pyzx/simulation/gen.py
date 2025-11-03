import pyzx as zx
from pyzx.utils import VertexType
from pyzx.simulation.circs import strCirc
from pyzx.simulation.stab import find_stab
from functools import cache


chars = [f"a{i}" for i in range(1000)]


def apply_effect(g: zx.Graph, vertex_types: str, phases: list[str]) -> None:
    outputs = g.outputs()
    if len(vertex_types) > len(outputs):
        raise TypeError("Too many output effects specified")
    new_outputs = []
    for i in range(len(vertex_types)):
        v = outputs[i]
        if vertex_types[i] == "/":
            new_outputs.append(v)
            continue
        if vertex_types[i].lower() in ("x", "z"):
            g.scalar.add_power(-1)
            g.set_type(v, VertexType.X)
            g.set_phase(v, phases[i])
        else:
            raise TypeError(
                f"Unknown vertex type {vertex_types[i]}. Only 'X' and 'Z' are allowed."
            )
    g.set_outputs(tuple(new_outputs))


def apply_state(g: zx.Graph, vertex_types: str, phases: list[str]) -> None:
    inputs = g.inputs()
    if len(vertex_types) > len(inputs):
        raise TypeError("Too many input states specified")
    new_inputs = []
    for i in range(len(vertex_types)):
        v = inputs[i]
        if vertex_types[i] == "/":
            new_inputs.append(v)
            continue
        if vertex_types[i].lower() in ("x", "z"):
            g.scalar.add_power(-1)
            g.set_type(v, VertexType.X)
            g.set_phase(v, phases[i])
        else:
            raise TypeError(
                f"Unknown vertex type {vertex_types[i]}. Only 'X' and 'Z' are allowed."
            )
    g.set_inputs(tuple(new_inputs))


def make_parametrized(g: zx.Graph, n_params: int):
    n_outputs = g.num_outputs()
    n_inputs = g.num_inputs()
    g = g.copy()
    strEffect = "x" * n_params + "/" * (n_outputs - n_params)
    g_adj = g.adjoint()

    if n_inputs > 0:
        g.apply_state("0" * n_inputs)
    apply_effect(g, strEffect, [chars[i] for i in range(n_params)])

    if n_inputs > 0:
        g_adj.apply_effect("0" * n_outputs)
    apply_state(g_adj, strEffect, [chars[i] for i in range(n_params)])

    g.compose(g_adj)
    return g


def get_g_list(g: zx.Graph):
    zx.full_reduce(g, paramSafe=True)
    g.normalize()
    g_list = find_stab(g)
    return g_list


@cache
def make_g_list(n_params: int):
    c = zx.qasm(strCirc)
    n = c.qubits
    g = c.to_graph()
    g.normalize()
    # zx.draw(g)

    zx.full_reduce(g)
    g.normalize()

    # strQ = "qreg q[" + str(n) + "];"
    # for i in range(n):
    #     strQ = strQ + "rz(0) q[" + str(i) + "];"
    # gGreen = zx.qasm(strQ).to_graph()
    # g.compose(gGreen)
    # gGreen.compose(g)
    # g = gGreen
    # zx.draw(g)
    # print("T-count = ", zx.tcount(g))

    NQ = n_params

    g = g.copy()
    h = g.copy()

    strEffect = "x" * NQ + "/" * (n - NQ)
    g_adj = g.adjoint()

    g.apply_state("0" * n)
    apply_effect(g, strEffect, [chars[i] for i in range(NQ)])

    g_adj.apply_effect("0" * n)
    apply_state(g_adj, strEffect, [chars[i] for i in range(NQ)])

    g.compose(g_adj)
    # zx.draw(g, scale=20, labels=False)
    zx.full_reduce(g, paramSafe=True)
    g.normalize()
    # zx.draw(g, scale=20, labels=False)

    # print("T-count = ", zx.tcount(g))

    g_list = find_stab(g)

    return g_list


# h.apply_state("0" * n)
# h.apply_effect("10100" + "/" * (n - 5))
# h.compose(h.adjoint())
# zx.full_reduce(h)
# h.normalize()
# h_list = find_stab(h)
# result_exact = sum(h_i.scalar.to_number() for h_i in h_list)
# print(result_exact)
