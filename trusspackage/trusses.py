import numpy as np
import matplotlib.pyplot as plt

""" 
    
    2D trusses only in this file.
    
    computes behaviour of a truss by the 'direct stiffness' method, which looks like 
    treating each member of the truss like an element in a FEM calculation. 
    
    Global coordinates: for N nodes, there are 2N coordinates. Nodes are ordered, and displacement 
    specified by [node_1_x, node_1_y, node_2_x, node_2_y, ... ]. Force is specified by an equivalent vector. 
    
    Local coordinates. Each member connects two nodes: nodeA and nodeB. Local coordinates
    for a member is 4 dimensional, and described by [node_A_x, node_A_y, node_B_x, node_B_y].
    
    each member has a stiffness matrix which enforces a relation between its local coordinates. 
    
    the calculation goes through the members and inserts the local stiffness matrix into a global 2N*2N
    matrix to determine the response of the whole system. As it does so, it scales each matrix to the stiffness of the 
    member. 
    
    Result is the global stiffness matrix:       f = K d. 
    
    some elements of d are known as dirichlet Boundary conditions. The complimentary elements of f are known as neumann BCs.
    These BCs are applied by removing and rearranging rows and cols of K, f and d.
    Output is the a full d and f vectors representing the unknown displacements and any reaction forces.             
    
    """


class Node:
    # node in a truss
    def __init__(self, pos, index):
        self.index = index
        self.pos = np.asarray(pos)       # nominal position
        self.disp_pos = np.asarray(pos)  # displaced position


class Member:
    # stick between two nodes
    def __init__(self, nodeA, nodeB):
        self.nodeA = nodeA
        self.nodeB = nodeB

        # construct the stiffness matrix relative to local coordinates.
        # B is stiffness in local coordinates.
        space_vec = nodeA.pos - nodeB.pos
        es_in_space = np.vstack((np.eye(2), - np.eye(2)))
        v = es_in_space.dot(space_vec)
        self.B = np.outer(v, v)


class Geometry:
    def __init__(self):
        self.nodes = []
        self.members = []

    def addnode(self, pos):
        self.nodes.append(Node(pos, len(self.nodes)))

    def addmember(self, nodeAindex, nodeBindex):
        self.members.append(Member(self.nodes[nodeAindex], self.nodes[nodeBindex]))

    def _mapping(self, element):
        # returns a local to global degree of freedom map as a list
        na = element.nodeA.index
        nb = element.nodeB.index
        return [na * 2, na * 2 + 1, nb * 2, nb * 2 + 1]  # mapping convention used here

    def mapB(self, element):
        # embeds an element's B matrix into a 2*n by 2*n matrix also called B...
        n = len(self.nodes)
        map_ = lambda i: self._mapping(element)[i]
        B = np.zeros((2 * n, 2 * n))
        for i in range(4):
            for j in range(4):
                B[map_(i), map_(j)] = element.B[i, j]
        return B

    """ ============== drawing ================ """

    def draw(self, displacements=np.asarray([]), hold=False):
        # draw structure
        if displacements.size == 0:
            displacements = np.zeros(2 * len(self.nodes))

        for node in self.nodes:
            disp = np.array([displacements[node.index * 2], displacements[node.index * 2 + 1]])
            node.disp_pos = node.pos + disp

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(121)
        x = [node.disp_pos[0] for node in self.nodes]
        y = [node.disp_pos[1] for node in self.nodes]

        indices = [node.index for node in self.nodes]
        plt.scatter(x, y)
        for i, txt in enumerate(indices):
            plt.annotate(txt, (x[i] + 0.1, y[i] - 0.05))

        for i, member in enumerate(self.members):
            a = member.nodeA.disp_pos
            b = member.nodeB.disp_pos
            X = np.asarray([a, b]).T
            lab = 'member ' + str(i)
            plt.plot(*X, label=lab)

        # rescale plot
        plt.axis('square')
        xmin, xmax, ymin, ymax = plt.axis()
        plt.xlim((xmin - 0.5, xmax + 0.5))
        plt.ylim((ymin - 0.5, ymax + 0.5))
        plt.legend(bbox_to_anchor=(1.02, 0., 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
        if not hold:
            plt.show()

    def plot_vector(self, vector):
        # draw any global vector in real space with quivers
        self.draw(hold=True)
        XY = np.asarray([node.pos for node in self.nodes])
        UV = np.reshape(vector, np.array([vector.shape[0] / 2, 2]).astype(int))

        plt.quiver(XY[:, 0], XY[:, 1], UV[:, 0], UV[:, 1], scale=1, scale_units='xy')
        plt.show()


class Model:
    def __init__(self, geom):
        self.geom = geom

        self.Bs = []

        # shouldn't be changing the structure once its been made into a model rather than a geometry
        # k is the dimension of global degree of freedom vectors.
        self.n = len(geom.nodes)
        self.k = 2 * self.n
        self.m = len(geom.members)

        # structure is fixed, so compute all the Bs straight away.
        for member in self.geom.members:
            # global coordinates.
            B = geom.mapB(member)
            self.Bs.append(B)

    def makeK(self, b):
        # b is a vector of stiffnesses for the members.
        # build the stiffness matrix as    K = sum_{members} stiffness_i * B_i
        assert (len(b) == self.m)
        K = np.zeros((self.k, self.k))
        for i, Bi in enumerate(self.Bs):
            K += b[i] * Bi
        return K

    def solve_for(self, b, f, d):
        """ solves the statics for a given set of boundary conditions and stiffness vector.
        b is the stiffness vector, in the ordered in the same way as the members in the geometry.
        f is the force boundary conditions
        d is displacement boundary conditions.

        all degrees of freedom must have either a force or displacement boundary condition.
        e.g.
            d = [a, b, -, c, -, -, -]
            f = [-, -, x, -, y, z, w]

        This function assumes by default that a displacement BC is set:

        d must contain float or None. If an element is None, then this is taken to mea that this dof has
        a force BC. So all elements of f are ignored except those for which the corresponding element of
        d is zero.
        e.g.
            d = [1., 0.5, None, 0.0, None, None, None]
            f = [0., 0.,  0.,   0.,  0.,   0.,   1.0 ]

            then f[0], f[1], f[3] are meaningless and are ignored.
        """

        # solve
        rows_to_go = []
        rows_to_stay = []
        dd = d.copy()
        for i, val in enumerate(d):
            if val is not None:
                rows_to_go.append(i)
            else:
                rows_to_stay.append(i)
                dd[i] = 0

        # displacement BCs
        K = self.makeK(b)
        fprime = f - K.dot(dd)

        Kprime = np.delete(K, rows_to_go, 0)
        Kprime = np.delete(Kprime, rows_to_go, 1)
        fprime = np.delete(fprime, rows_to_go)
        dprime = list(np.linalg.solve(Kprime, fprime))

        dout = dd.copy()
        for i in range(len(dprime)):
            dout[rows_to_stay[i]] = dprime[i]
        dout = np.array(dout)

        # return the completed f and d.
        return K.dot(dout), dout

    def invert_determinate(self, d_in, f_in):
        # make V matrix
        # this can invert for a determine truss, but otherwise will run but return nonsense.
        d = d_in  # np.delete(d_in, [0,1,3,5,7])
        f = np.delete(f_in, [0, 1, 3, 5, 7])
        V = []
        for B_matrix in self.Bs:
            B = np.delete(B_matrix, [0, 1, 3, 5, 7], axis=0)
            B = np.delete(B, [0, 1, 3, 5, 7], axis=1)
            lam, vecs = np.linalg.eig(B)
            vecs = vecs.T
            for j in range(2):
                if abs(lam[j]) > 0.0000001:
                    V.append(vecs[j])
                    break
        V[2] = V[2] * np.sqrt(2)
        V[1] = V[1] * np.sqrt(2)
        V = np.asarray(V).T
        C = np.linalg.inv(V).T

        f_i = np.dot(C.T, f)
        gamma_i = np.dot(V.T, d)
        bb = np.divide(f_i, gamma_i)

        return bb

    def drawdisp(self, d):
        # draw the deformed structure just for fun.
        self.geom.draw(d)
