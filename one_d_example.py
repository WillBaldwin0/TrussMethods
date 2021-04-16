from trusspackage.trusses import Model, Geometry
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D # needed

" ============================= make the truss and decide BC's ============================ "

# 4 nodes in a line
G = Geometry()
G.addnode([0, 0])
for i in range(3):
    G.addnode([i + 1, 0])
    G.addmember(i, i + 1)
S = Model(G)

# show what it looks like
# G.draw()

# make 1d by fixing the y displacement of all nodes to be 0
d_BC = [0, 0, None, 0, None, 0, None, 0]
f = np.array([0, 0, 0, 0, 1, 0, 0.05, 0])


" ============================= Define Omega ============================ "

# define Omega through vectors and coefficients
# e1 and e2 are the basis vectors of the model space.
# coeficients of these are the parameters.
# in this case model is linear variation. parameters (p) are stiffness of the first and last member (/100)
# parameter vectors are 2d.
# span of e1, e2 is a subspace of the three-dimensional space of  stiffness (b) vectors.
epsilon = 0.01
e1 = np.array([1, 0.5, 0]) / epsilon
e2 = np.array([0, 0.5, 1]) / epsilon

# make a set of stiffness vectors (b) that lie on the plane.
# in this case we let all p vary between 0 and 0.6, arbitrary decision compatible with the e1, e2 and measured dt vector.
Onum = 150
max_p_i = 0.6
p_range = np.linspace(0.001, max_p_i, Onum)
p_list = []
b_list = []

# make a list of the points in p-space and b-space.
for i in range(Onum):
    for j in range(Onum):
        b = p_range[i] * e1 + p_range[j] * e2
        if all(thing > 0.1 for thing in b):
            b_list.append(b)
            p_list.append([p_range[i], p_range[j]])

b_arr = np.asarray(b_list)
p_arr = np.asarray(p_list)

" ============================= Image of Omega ============================ "

# collect lost of other data at the same time.
# make a list of d's by computing the displacement for every p in Omega (every b  phi(Omega))
# let x = dt - d be the difference between the predicted d at p, alpha(p), and the measured d denoted dt.
# from now on we only care about the 'free' degrees of freedom which are 2,4 and 6.
# so d now refers to the 3d vector which we care about.
# let val bu the likelihood of p which depends only on x.

# for making plots, choose a 'measurement' dt, and setup a likelihood function to use.
dt = np.array([0.1,0.17,0.23])

# for the likelihood.
likelihood_variance = 0.01
def f_sig(err, sig):
    return np.exp(-sig ** (-2) * err.dot(err) / 2) / (2 * np.pi * sig ** 2) ** (err.shape[0] / 2)

x_list = []
d_list = []
val_list = []

# go through the b's and compute d, x, val
for b in b_list:
    f_, d_ = S.solve_for(b, f, d_BC)
    # extract the 3 degrees of freedom we care about.
    d = np.delete(d_, [0, 1, 3, 5, 7])
    d_list.append(d)
    x_list.append(d - dt)
    val_list.append(f_sig(d - dt, likelihood_variance))

d_arr = np.asarray(d_list)
x_arr = np.asarray(x_list)
val_arr = np.asarray(val_list)

# trim all lists down for graphing with some mask
# we end up plotting a subset of alpha(omega)
remove_points = [index for index, d in enumerate(d_arr) if (np.linalg.norm(d) > 1.4 * np.linalg.norm(dt) or np.linalg.norm(d) < 0.5 * np.linalg.norm(dt))]
print(d_arr.shape[0] - len(remove_points))
d_arr = np.delete(d_arr, remove_points, axis=0)
x_arr = np.delete(x_arr, remove_points, axis=0)
p_arr = np.delete(p_arr, remove_points, axis=0)
b_arr = np.delete(b_arr, remove_points, axis=0)
val_arr = np.delete(val_arr, remove_points)

" ============================= Visualisation ============================ "

# plot with just scattered points
if True:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(d_arr[:, 0], d_arr[:, 1], d_arr[:, 2], c=val_arr)
    ax.scatter(dt[0], dt[1], dt[2], c='r', s=30, alpha=1)
    ax.set_xlabel('d.o.f. 2')
    ax.set_ylabel('d.o.f. 4')
    ax.set_zlabel('d.o.f. 6')
    plt.show()

