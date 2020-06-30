from trusspackage.trusses import Geometry, Model
from trusspackage.templates import det_tower, indet_tower
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

G=det_tower(2,2)
S=Model(G)
G.draw()

# set bcs
d=[None]*S.k
f=np.zeros(S.k)
d[0]=0
d[1]=0
d[2]=0
d[3]=0
f[4]=1

# create a plane of displacements
epsilon=0.01
e1=np.array([1,epsilon,1,1])/epsilon
e2=np.array([1,1,epsilon,1])/epsilon

num=175
coefs = np.linspace(-1,1,num)
b_list = []

for i in range(num):
    for j in range(num):
        b=coefs[i]*e1+coefs[j]*e2
        if all(i > 10 for i in b): #ensure all stiffness > 0
            b_list.append(b)

# map them through the structure
dr_list=[]

f_sig = lambda x, sig : np.exp(-sig**(-2) * x.dot(x) / 2) / (2*np.pi*sig**2)**(x.shape[0]/2)

drt=np.array([0.1,0.02,-0.065])
val_list=[]
x_list=[]
min=10000
stn=0

for b in b_list:
    f_, d_ = S.solve_for(b,f,d)
    # this d_ of ds includes the zero boundary conditions as the first 4 elements, so get rid of them for a start
    # now we have 4 remaining degrees of freedom. lets forget the second one, P removes the second degree of freedom
    dr = np.delete(d_, [0,1,2,3,5])
    xx=dr-drt
    if np.linalg.norm(xx)<min:
        min=np.linalg.norm(xx)
        stn=min/np.linalg.norm(dr)
    val=f_sig(xx,0.03)
    dr_list.append(dr)
    val_list.append(val)

print(stn)
print(val_list)
# now make them into xyz lists
dr_list = np.asarray(dr_list)
x=dr_list[:,0]
y=dr_list[:,1]
z=dr_list[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,c=val_list)
ax.scatter(drt[0], drt[1], drt[2], s=100)
ax.set_xlabel('d.o.f. 4')
ax.set_ylabel('d.o.f. 6')
ax.set_zlabel('d.o.f. 7')
plt.show()






