from trusspackage.trusses2d2 import Model, Geometry
import numpy as np

# make truss: 4 nodes in a line
G = Geometry()

G.addnode([0, 0]) # node index 0
G.addnode([1, 0]) # node index 1
G.addnode([0, 1]) # ...
G.addnode([1, 1])

G.addmember(0, 2) # member index 0
G.addmember(1, 2)
G.addmember(1, 3)
G.addmember(2, 3)

# show what it looks like, and to see the node and member indexing scheme.
G.draw()

# make calculator
S = Model(G)

# bounary conditions
# fix the bottom two nodes, let the top two be free
d_BC = [0, 0, 0, 0, None, None, None, None]

# must have a force BC on every degree of freedom that is not constrained by a displacement BC.
# supply a full vector, and the the force BCs corresponding to the degrees of freedom that do have
# displacement BCs are ignored

# force of 0.2 to the right on the upper right hand node:
f = np.array([0, 0, 0, 0, 0, 0, 0.2, 0])
# first four elements are ignored, the last four set the force on the upper nodes.


# the drawing function shows the order of the members.
# make stiffness vector where members 0 through 3 have stiffnesses [1,2,3,4]
b = np.array([1,2,3,4])

# solve. function returns info by filling in the d and f vectors.
ff, dd = S.solve_for(b, f, d_BC)

print(ff)
print(dd)

# draw it.
S.drawdisp(dd)
