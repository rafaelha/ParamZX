# %%
from collections import defaultdict
import numpy as np
import pyzx as zx
from pyzx.utils import VertexType
import math
import cmath
from pyzx.simulation.circs import strCirc
from pyzx.simulation.stab import find_stab

size = 10

c = zx.qasm(strCirc)
n = c.qubits
g = c.to_graph()
g.normalize()
zx.draw(g)

zx.full_reduce(g)
g.normalize()


strQ = "qreg q[" + str(n) + "];"
for i in range(n):
    strQ = strQ + "rz(0) q[" + str(i) + "];"
gGreen = zx.qasm(strQ).to_graph()
g.compose(gGreen)
gGreen.compose(g)
g = gGreen
zx.draw(g)
print("T-count = ", zx.tcount(g))

# %%


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


chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
NQ = 5
n_params = NQ

g = g.copy()
h = g.copy()

strEffect = "x" * NQ + "/" * (n - NQ)
g_adj = g.adjoint()

g.apply_state("0" * n)
apply_effect(g, strEffect, [chars[i] for i in range(NQ)])

g_adj.apply_effect("0" * n)
apply_state(g_adj, strEffect, [chars[i] for i in range(NQ)])

g.compose(g_adj)
zx.draw(g, scale=20, labels=False)
zx.full_reduce(g, paramSafe=True)
g.normalize()
zx.draw(g, scale=20, labels=False)

print("T-count = ", zx.tcount(g))

g_list = find_stab(g)
print(len(g_list))

#%%

# h.apply_state("0" * n)
# h.apply_effect("10100" + "/" * (n - 5))
# h.compose(h.adjoint())
# zx.full_reduce(h)
# h.normalize()
# h_list = find_stab(h)
# result_exact = sum(h_i.scalar.to_number() for h_i in h_list)
# print(result_exact)

# %%

print("Number of terms per graph...")
print("(1+e^x)/e^((1/2)x)\t (e^(x*y))\t phPair")
# node (1+e^x) - scalar.phasenodes
# B    e^x/2   - scalar.phasevars_halfpi[1] [3]
# C    e^(x*y) - scalar.phasevars_pi_pair
# D    (1+e^x+e^y+e^(x+y)) - scalar.phasepairs

data = []
for i in range(len(g_list)):
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

for i in range(len(g_list)):
    g_i = g_list[i]
    for term in range(len(g_i.scalar.phasenodevars)):
        bitstr = [0] * n_params
        for v in g_i.scalar.phasenodevars[term]:
            bitstr[char_to_idx[v]] = 1
        constTerm = int(g_i.scalar.phasenodes[term] * 4)

        rowData = []
        rowData.append(i)  # Graph_id
        rowData.append(4)  # type: node type = 4 (as this is a multiplier, 4/4 = 1x)
        rowData.append(constTerm)  # const
        rowData.extend(bitstr)
        compiled_ab.append(rowData)

    for j in [1, 3]:
        for term in range(len(g_i.scalar.phasevars_halfpi[j])):
            bitstr = [0] * n_params
            for v in g_i.scalar.phasevars_halfpi[j][term]:
                bitstr[char_to_idx[v]] = 1
            constTerm = 0
            # (1/2)*4 = 2 or (3/2)*4 = 6 - this (divided by 4) is a multiplier
            ttype = int((j / 2) * 4)

            rowData = []
            rowData.append(i)  # Graph_id
            rowData.append(ttype)  # type
            rowData.append(constTerm)  # const
            rowData.extend(bitstr)
            compiled_ab.append(rowData)


compiled_ab = np.array(compiled_ab)

char_to_idx = {char: i + 1 for i, char in enumerate(chars)}
char_to_idx["1"] = 0
n_params = NQ + 1
compiled_c = []

for i in range(len(g_list)):
    graph = g_list[i].copy()
    
    for pSet in graph.scalar.phasevars_pi_pair:
        bitstr = [0] * n_params*2
        for p in pSet[0]:
            bitstr[char_to_idx[p]] = 1
        for p in pSet[1]:
            bitstr[char_to_idx[p] + n_params] = 1

        rowData = []
        rowData.append(i) # Graph_id
        rowData.extend(bitstr)
        compiled_c.append(rowData)
        
compiled_c = np.array(compiled_c)

n_ancil = 3 # Extra bits needed: Multiplier, const term alpha, const term beta
compiled_d = []
char_to_idx = {char: i for i, char in enumerate(chars)}
n_params = NQ

for i in range(len(g_list)):
    graph = g_list[i]
    for pp in range(len(graph.scalar.phasepairs)):
        bitstr = [0] * n_params*2
        for v in graph.scalar.phasepairs[pp].paramsA:
            bitstr[char_to_idx[v]] = 1
        for v in graph.scalar.phasepairs[pp].paramsB:
            bitstr[char_to_idx[v] + n_params] = 1
        constTermA = int(graph.scalar.phasepairs[pp].alpha)
        constTermB = int(graph.scalar.phasepairs[pp].beta)

        rowData = []
        rowData.append(i) # Graph_id
        rowData.append(constTermA) # const term alpha
        rowData.append(constTermB) # const term beta
        rowData.extend(bitstr)
        compiled_d.append(rowData)

compiled_d = np.array(compiled_d)


compiled_static = []

for i in range(len(g_list)):
    graph = g_list[i]
    rowData = []
    rowData.append(int(graph.scalar.phase*4))      # phase
    rowData.append(graph.scalar.power2)            # power2
    rowData.append(graph.scalar.floatfactor.real)  # float factor (real part)
    rowData.append(graph.scalar.floatfactor.imag)  # float factor (imag part)
    compiled_static.append(rowData)

compiled_static = np.array(compiled_static)

#%%  EXECUTION ------------------
# From this data, calculate the final result... (node-type terms)
paramVals = [1,0,1,0,0]  # values of the parameters

# AB

inp = np.array([1,1,1] + paramVals)
termVals = [0.0+0.0j]*len(compiled_ab)

compiled_ab = compiled_ab * inp
    
# Calculate each row:
for i in range(len(compiled_ab)):
    row = compiled_ab[i]
    rowsum = sum(row[3:])%2
    const = row[2]/4
    ttype = row[1]/4
    phase = (rowsum+const)%2
    phase *= ttype
    phase = int(phase*4)
    
    expTerm = 0.0 + 0.0j
    match phase:
        case 0: 
            expTerm = 1
        case 1:
            expTerm = 0.7071067811865 + 0.7071067811865j
        case 2:
            expTerm = 1j
        case 3:
            expTerm = -0.7071067811865 + 0.7071067811865j
        case 4:
            expTerm = -1
        case 5:
            expTerm = -0.7071067811865 - 0.7071067811865j
        case 6:
            expTerm = -1j
        case 7:
            expTerm = 0.7071067811865 - 0.7071067811865j  
    if ttype == 1:
        expTerm += ttype # TODO: fix double usage of ttype
    termVals[i] = expTerm

summands = defaultdict(lambda: 1+0j)
for i in range(len(compiled_ab)):
    graph_id = compiled_ab[i][0]
    term = termVals[i]
    summands[graph_id] *= term
        
summands_ab = summands

# C

inp = np.array([1, 1] + paramVals + [1] + paramVals)
termVals = [0.0+0.0j]*len(compiled_c)

compiled_c = compiled_c * inp
    
# Calculate each row:
for i in range(len(compiled_c)):
    row = compiled_c[i]
    rowsumA = sum(row[1:n_params+1])%2
    rowsumB = sum(row[n_params:])%2
    rowsumProd = rowsumA*rowsumB # rowsumA AND rowsumB
    
    expVal = 1  # exp(i*pi*0) =  1
    if (rowsumProd == 1):
        expVal = -1  # exp(i*pi*1) = -1
    termVals[i] = expVal
    
summands = defaultdict(lambda: 1+0j)
for i in range(len(compiled_c)):
    graph_id = compiled_c[i][0]
    term = termVals[i]
    summands[graph_id] *= term
        
summands_c = summands

# D

def cexp(val) -> complex:
    return cmath.exp(1j*math.pi*val)

inp = [1,1,1] + (paramVals*2)
termVals = [0.0+0.0j]*len(compiled_d)

compiled_d = compiled_d * inp
n_ancil = 3
    
# Calculate each row:
for i in range(len(compiled_d)):
    row = compiled_d[i]
    rowsumA = sum(row[n_ancil:n_params+n_ancil])%2
    rowsumB = sum(row[n_params+n_ancil:])%2
    alpha = (row[1] + (rowsumA*4)) % 8
    beta  = (row[2] + (rowsumB*4)) % 8
    
    gamma = (alpha+beta) % 8
    
    expTerm = 1.0 + cexp(alpha/4) + cexp(beta/4) - cexp(gamma/4)
    termVals[i] = expTerm
    

summands = defaultdict(lambda: 1+0j)
for i in range(len(compiled_d)):
    graph_id = compiled_d[i][0]
    term = termVals[i]
    summands[graph_id] *= term
        
summands_d = summands

#%% RESULT ------------------

root2 = 1.4142135623730950488
result = 0+0j
for g_id in range(len(g_list)):
    v = summands_ab[g_id] * summands_c[g_id] * summands_d[g_id]
    v *= cexp(g_list[g_id].scalar.phase)
    v *= root2**g_list[g_id].scalar.power2
    v *= g_list[g_id].scalar.floatfactor
    result += v

assert np.isclose(result, -1.6080344521218759e-15+0.0683018803681063j)
