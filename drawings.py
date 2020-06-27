from trusspackage.trusses import Geometry, Model
from trusspackage.templates import det_tower, indet_tower
import numpy as np

G=indet_tower(16,4)
S=Model(G)

b=np.ones(S.m)
d=[None]*S.k
f=np.zeros(S.k)

for i in range(4):
    d[2*i+1]=0
    d[2*i]=0
    f[S.k-2*i-1]=1

f, d = S.solve_for(b,f,d)
S.drawdisp(d)
