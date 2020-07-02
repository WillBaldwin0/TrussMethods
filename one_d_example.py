from trusspackage.trusses import Model, Geometry
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

G=Geometry()
G.addnode([0,0])

for i in range(3):
    G.addnode([i+1,0])
    G.addmember(i, i+1)
S=Model(G)

# can use as 1d if we just fix all the vertical degrees of freedom.
d = [0,0,None,0,None,0,None,0]
f = np.zeros(S.k)
f[6]=1

# create a list of b-vectors that are all in the plane spanned by e1 and e2
epsilon = 0.001
e1 = np.array([1,epsilon,1])/epsilon
e2 = np.array([1,1,epsilon])/epsilon

num = 175       # O(num^2) vectors
coefs = np.linspace(-1,1,num)
coef_list = []  # for the final plot
b_list = []

for i in range(num):
    for j in range(num):
        b=coefs[i]*e1+coefs[j]*e2
        if all(i > 10 for i in b):      # ensure all stiffness > 0. (10 is 'small')
            b_list.append(b)
            coef_list.append([i,j])

# map them through the structure K^{-1} f. Also make f_sig and a measured value drt.
f_sig = lambda err, sig: np.exp(-sig**(-2) * err.dot(err) / 2) / (2*np.pi*sig**2)**(err.shape[0]/2)
drt = np.array([0.01,0.04,0.12])

dr_list = []
val_list = []
x_list = []     # x = dr - drt
min_x = 10000   # track the smallest |x|
stn = 0         # minimum signal-to-noise ratio

for b in b_list:
    f_, d_ = S.solve_for(b,f,d)
    # d and d_ include the zero boundary conditions as the first 4 elements, so get rid of them
    # Also remove d[5], this is 'P'. Don't need to do this other than for visualisation.
    dr = np.delete(d_, [0,1,3,5,7])
    x = dr-drt
    val = f_sig(x,0.03)
    dr_list.append(dr)
    val_list.append(val)

    # tracking minimum
    if np.linalg.norm(x) < min_x:
        min_x = np.linalg.norm(x)
        stn = min_x/np.linalg.norm(dr)

# print the minimum signal to noise ratio required to get this data assuming the parametrisation is accurate
print(stn)

# make xyz lists to plot
dr_list = np.asarray(dr_list)
x = dr_list[:,0]
y = dr_list[:,1]
z = dr_list[:,2]

# plot a surface to show the curvature and the likelihoods
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,c=val_list)
ax.scatter(drt[0], drt[1], drt[2], s=100)   # plot the drt point too.
ax.set_xlabel('d.o.f. 4')
ax.set_ylabel('d.o.f. 6')
ax.set_zlabel('d.o.f. 7')
plt.show()

# plot the likelihoods over the b plane
coef_list = np.array(coef_list)
x = coef_list[:,0]
y = coef_list[:,1]

# easier to just scatter coloured dots because of a possibly incomplete grid of data
plt.scatter(x,y,c=val_list)
plt.show()
f[8] = 1



