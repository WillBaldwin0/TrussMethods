from trusspackage.trusses import Model, Geometry
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


" ============================= make the truss and decide BC's ============================ "

G = Geometry()
G.addnode([0,0])
for i in range(3):
    G.addnode([i+1,0])
    G.addmember(i, i+1)
S = Model(G)

d_BC = [0,0,None,0,None,0,None,0]
f = np.array([0,0,0,0,1,0,0.05,0])



" ============================= Define and parameterise Omega ============================ "

e1 = np.array([1,0.5,0])
e2 = np.array([0,0.5,1])

# make points in a circle, with distributed uniformly with respect to 1/r and theta.

def invspace(lower, upper, num):
    # evenly spaced reciporicals
    u = 1/upper
    l = 1/lower
    recips = np.linspace(u,l,num)
    return np.power(recips, -1)[::-1]

# find orthogonal vectors in the plane of e1, e2
f1 = np.cross(np.cross(e1,e2),e1)
f1 = f1/np.linalg.norm(f1)
f2 = e1/np.linalg.norm(e1)

# make a set of nicely spaced bs that lie on the plane.
rad_num = 100
min_rad = 8
max_rad = 23
radii = invspace(min_rad, max_rad, rad_num)
theta_num = 100
thetas = np.linspace(0, 1.53, theta_num)

b_list = []
p_list = []
# b = [ p_1 , (p_1+p_2)/2 , p2 ]

for i in range(rad_num):
    for j in range(theta_num):
        b = radii[i]*(f1*np.cos(thetas[j]) + f2*np.sin(thetas[j]))
        if all(thing > 0.1 for thing in b):
            b_list.append(b)
            p_list.append(np.linalg.pinv(np.vstack((e1,e2)).T).dot(b)) # project the b's back onto the e1, e2 basis.

b_arr = np.asarray(b_list)
p_arr = np.asarray(p_list)

" ============================= Image of Omega ============================ "

# choose a measurement.
dt = np.array([0.1,0.17,0.23])


# choose a likelihood func.
likelihood_variance = 0.01
def f_sig(err, sig):
    return np.exp(-sig**(-2) * err.dot(err) / 2) / (2*np.pi*sig**2)**(err.shape[0]/2)


x_list = []
d_list = []
val_list = []

# image of forward problem
# again, only care about the horizontal motion of the three right most nodes, which are dofs 2, 4 and 6.
for b in b_list:
    f_, d_ = S.solve_for(b,f,d_BC)
    d = np.delete(d_, [0,1,3,5,7])
    d_list.append(d)
    val_list.append(f_sig(d - dt, likelihood_variance))

d_arr = np.asarray(d_list)
val_arr = np.asarray(val_list)


# trim all lists down for graphing with some mask or set of masks:
remove_points = [index for index, d in enumerate(d_arr) if np.linalg.norm(d) > 0.4]
print('using {} points'.format(d_arr.shape[0]-len(remove_points)))
d_arr = np.delete(d_arr, remove_points, axis = 0)
p_arr = np.delete(p_arr, remove_points, axis = 0)
b_arr = np.delete(b_arr, remove_points, axis = 0)
val_arr = np.delete(val_arr, remove_points)


" ============================= Visualisation ============================ "


# plot surface by scattering points, and colour them by likelihood.
if True:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(d_arr[:,0], d_arr[:,1], d_arr[:,2], c=val_arr)
    ax.scatter(dt[0],dt[1],dt[2],c='r')

    # for axis limits
    max_disp = max([max(d) for d in d_arr])
    ax.set_xlim3d(0,max_disp)
    ax.set_ylim3d(0,max_disp)
    ax.set_zlim3d(0,max_disp)
    ax.set_xlabel('node 1')
    ax.set_ylabel('node 2')
    ax.set_zlabel('node 3')
    plt.show()


# plot a surface to show the curvature and the likelihoods with a sooth surface
if True:
    # create a triangulation for plotting
    input_triangulation = mtri.Triangulation(p_arr[:,0], p_arr[:,1])
    output_triangulation = mtri.Triangulation(d_arr[:,0], d_arr[:,1], triangles=input_triangulation.triangles)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # colours don't work for trisurf. Plots were made by saving the arrays and then running
    # the same plotting script in matlab where you can apply separate colours a surface.
    ax.plot_trisurf(d_arr[:,0], d_arr[:,1], d_arr[:,2], triangles=input_triangulation.triangles)
    ax.scatter(dt[0],dt[1],dt[2],c='r')

    # for axis limits
    max_disp = max([max(d) for d in d_arr])
    ax.set_xlim3d(0,max_disp)
    ax.set_ylim3d(0,max_disp)
    ax.set_zlim3d(0,max_disp)
    ax.set_xlabel('node 1')
    ax.set_ylabel('node 2')
    ax.set_zlabel('node 3')
    plt.show()


# plot the data in p space
if True:
    plt.scatter(p_arr[:,0],p_arr[:,1],c = val_arr)
    plt.show()
