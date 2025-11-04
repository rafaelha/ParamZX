#%%
from pyzx.simulation.msd import msd_graph
from pyzx.simulation.sampler import Sampler
from pyzx.simulation.tictoc import tic, toc
import pyzx as zx

g = msd_graph(17)
zx.draw(g)
# %%

tic()
sampler = Sampler(g)
toc("Compiled sampler in")
print(sampler)

# %%

s = sampler.sample(250, 100)

for row in s:
    print("".join(map(str, row)))
