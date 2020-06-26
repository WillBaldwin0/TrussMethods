from trusspackage.trusses import Geometry, Model
import numpy as np

T=Geometry()

T.addnode([0,0])
T.addnode([1,0])
for i in range(5):
    T.addnode([0,i+1]) # nodeindex 2(i+1)
    T.addnode([1,i+1]) # nodeindex 2(i+2)
    T.addmember(2*i  ,2*i+2)
    T.addmember(2*i+1,2*i+3)
    T.addmember(2*i  ,2*i+3)
    #T.addmember(2*i+1  ,2*i+2)
    T.addmember(2*i+2,2*i+3)

S=Model(T)

b=np.ones(S.m)*100
f=np.zeros(S.k)

d=[None]*S.k
d[0]=0
d[1]=0
d[2]=0
d[3]=0
d[22]=0.5
d[18]=0.5

f, d = S.solve_for(b,f,d)
print(d)

S.drawdisp(d)
