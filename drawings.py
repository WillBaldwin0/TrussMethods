from trusspackage.trusses import Geometry, Model
from trusspackage.templates import det_tower, indet_tower
import numpy as np

width=3
G=indet_tower(8,width)
S=Model(G)

b=np.ones(S.m)
d=[None]*S.k
f=np.zeros(S.k)

for i in range(width):
    d[2*i+1]=0
    d[2*i]=0
    f[S.k-2*i-2]=0.01

f, d = S.solve_for(b,f,d)
S.drawdisp(d)
