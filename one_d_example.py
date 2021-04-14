from trusspackage.trusses import Model, Geometry
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

######################## System setup ############################################

G = Geometry()
G.addnode([0, 0])
for i in range(3):
    G.addnode([i + 1, 0])
    G.addmember(i, i + 1)
S = Model(G)

d_BC = [0, 0, None, 0, None, 0, None, 0]
f = np.array([0, 0, 0, 0, 1, 0, 0.05, 0])
dt = np.array([0.1, 0.2, 0.246])

likelihood_variance = 0.002


def f_sig(err, sig):
    return np.exp(-sig ** (-2) * err.dot(err) / 2) / (2 * np.pi * sig ** 2) ** (err.shape[0] / 2)


######################################## Make Omega ############################################
# there are many descriptions so we can do different visualisations

# define Omega through vectors and coefficients - these are K_n and p_n
epsilon = 0.01
e1 = np.array([1, 0.5, 0]) / epsilon
e2 = np.array([0, 0.5, 1]) / epsilon

# make a set of uniformly spaced bs that lie on the plane.
Onum = 200
max_p_i = 0.6
p_range = np.linspace(-max_p_i, max_p_i, Onum)
p_list = []
b_list = []

for i in range(Onum):
    for j in range(Onum):
        b = p_range[i] * e1 + p_range[j] * e2
        if all(thing > 0.1 for thing in b):
            b_list.append(b)
            p_list.append([p_range[i], p_range[j]])

b_arr = np.asarray(b_list)
p_arr = np.asarray(p_list)

################################### Image of Omega ###########################

x_list = []
d_list = []
val_list = []

for b in b_list:
    f_, d_ = S.solve_for(b, f, d_BC)
    d = np.delete(d_, [0, 1, 3, 5, 7])
    d_list.append(d)
    x_list.append(d - dt)
    val_list.append(f_sig(d - dt, likelihood_variance))

d_arr = np.asarray(d_list)
x_arr = np.asarray(x_list)
val_arr = np.asarray(val_list)

# trim all lists down for graphing with some mask or set of masks:
remove_points = [index for index, d in enumerate(d_arr) if np.linalg.norm(d) > 2 * np.linalg.norm(dt)]
print(d_arr.shape[0] - len(remove_points))
d_arr = np.delete(d_arr, remove_points, axis=0)
x_arr = np.delete(x_arr, remove_points, axis=0)
p_arr = np.delete(p_arr, remove_points, axis=0)
b_arr = np.delete(b_arr, remove_points, axis=0)
val_arr = np.delete(val_arr, remove_points)

########################### visulisation ##################################

# create a triangulation
tris = mtri.Triangulation(p_arr[:, 0], p_arr[:, 1]).triangles

# plot a surface to show the curvature and the likelihoods with a sooth surface
if True:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(d_arr[:, 0], d_arr[:, 1], d_arr[:, 2], triangles=tris)
    ax.scatter(dt[0], dt[1], dt[2], c='r')

    # for axis limits
    max_disp = max([max(d) for d in d_arr])
    ax.set_xlim3d(0, max_disp)
    ax.set_ylim3d(0, max_disp)
    ax.set_zlim3d(0, max_disp)
    ax.set_xlabel('node 1')
    ax.set_ylabel('node 2')
    ax.set_zlabel('node 3')
    plt.show()

# plot individual wires corresponding to constant p_n
if False:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the individual p1_wires
    p1_wires = []
    current_p = -100
    for index, p in enumerate(p_arr):
        if p[0] > current_p + 0.000001:
            current_p = p[0]
            p1_wires.append([d_arr[index]])
        else:
            p1_wires[-1].append(d_arr[index])

    for lst in p1_wires:
        arr = np.asarray(lst)
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2])

    # plot the individual p2_wires
    p2_wires = []
    current_p = -100
    for index, p in enumerate(p_arr):
        if p[1] > current_p + 0.000001:
            current_p = p[1]
            p2_wires.append([d_arr[index]])
        else:
            p2_wires[-1].append(d_arr[index])

    for lst in p2_wires:
        arr = np.asarray(lst)
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2])

    # measurment
    ax.scatter(dt[0], dt[1], dt[2], c='r')

    # for axis limits
    max_disp = max([max(d) for d in d_arr])
    ax.set_xlim3d(0, max_disp)
    ax.set_ylim3d(0, max_disp)
    ax.set_zlim3d(0, max_disp)
    ax.set_xlabel('node 1')
    ax.set_ylabel('node 2')
    ax.set_zlabel('node 3')
    plt.show()

# other one
P = np.array([[0, 1, 0], [1, 0, 1]])

if True:
    dr_arr = np.asarray(dr_list)
    vr_list = P.dot(dr_arr.T)
    print(vr_list.shape)
    plt.scatter(vr_list[0, :], vr_list[1, :], s=0.1)
    plt.show()

if True:
    used_bs = np.asarray(used_bs)
    x = used_bs[:, 0]
    y = used_bs[:, 1]
    z = used_bs[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=val_list_3d)
    ax.plot([b_true[0]], [b_true[1]], [b_true[2]], 'ro',
            markersize=3)  # plot the drt point too. s=100,, depthshade=False

    # for scaling
    a = max(used_bs[0])
    ax.scatter([0, a], [0, a], [0, a], s=0.1)

    ax.set_xlabel('d.o.f. 4')
    ax.set_ylabel('d.o.f. 6')
    ax.set_zlabel('d.o.f. 7')
    plt.show()

if False:
    b_list = np.asarray(b_list)
    x = b_list[:, 0]
    y = b_list[:, 1]
    z = b_list[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    """ax.plot([b_true[0]], [b_true[1]], [b_true[2]], 'ro', markersize =3) # plot the drt point too. s=100,, depthshade=False

    # for scaling
    a=max(used_bs[0])
    ax.scatter([0,a],[0,a],[0,a],s=0.1)

    ax.set_xlabel('d.o.f. 4')
    ax.set_ylabel('d.o.f. 6')
    ax.set_zlabel('d.o.f. 7')"""
    plt.show()

# plot the likelihoods over the b plane
print(np.array(coef_list).shape)
# coef_list = np.delete(np.array(coef_list), mess, axis=0)
coef_list = np.array(coef_list)
x = coef_list[:, 0]
y = coef_list[:, 1]

print(coefs[138] * e1 + coefs[110] * e2)

# easier to just scatter coloured dots because of a possibly incomplete grid of data
plt.scatter(x, y, c=val_list)
plt.show()

# calculate the mean b...
# add all te vectors and divite by the number of them...

number = sum(val_list_3d)
print(number)

used_bsT = used_bs.T
mean = used_bsT.dot(val_list_3d) / number
print(mean)

print()
print(np.linalg.norm(b_true - best_b))
print(np.linalg.norm(b_true - mean))
