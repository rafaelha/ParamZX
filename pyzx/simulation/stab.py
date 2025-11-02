import pyzx as zx
from typing import List
from pyzx.simulation.simplify import full_red

def find_stab(gg: zx.Graph, printOut: bool = False) -> List[zx.Graph]:
    if zx.simplify.tcount(gg) == 0:
        return [gg]
    gsum = zx.simulate.replace_magic_states(gg, False)

    full_red(gsum)
    output = []

    for hh in gsum.graphs:
        output.extend(find_stab(hh, printOut))

    if printOut:
        print(len(gsum.graphs), len(output))
    return output