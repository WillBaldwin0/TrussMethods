import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, pos, index):
        self.index=index
        self.pos = np.asarray(pos)
        self.disp_pos = np.asarray(pos)


class Member:
    def __init__(self, nodeA, nodeB):
        self.nodeA=nodeA
        self.nodeB=nodeB

        l=np.linalg.norm(nodeA.pos-nodeB.pos)
        cos=(nodeA.pos[0]-nodeB.pos[0])/l
        sin=(nodeA.pos[1]-nodeB.pos[1])/l
        v=np.array([cos,sin,-cos,-sin])
        self.B=np.outer(v,v)


class Geometry:
    def __init__(self):
        self.nodes=[]
        self.members=[]
        self.num_nodes=0
        self.m=0

    def addnode(self,pos):
        self.nodes.append(Node(pos, len(self.nodes)))
        self.num_nodes=len(self.nodes)

    def addmember(self,nodeAindex,nodeBindex):
        self.members.append(Member(self.nodes[nodeAindex], self.nodes[nodeBindex]))
        self.m=len(self.members)

    def _mapping(self, element):
        # returns a local to global degee of freedom map as a list
        na=element.nodeA.index
        nb=element.nodeB.index
        return [na*2,na*2+1,nb*2,nb*2+1] # mapping convention used here

    def mapB(self, element):
        # embeds an element's B matrix into a 2*n by 2*n matrix
        n=len(self.nodes)
        map_ = lambda i: self._mapping(element)[i]
        B=np.zeros((2*n,2*n))
        for i in range(4):
            for j in range(4):
                B[map_(i),map_(j)]=element.B[i,j]
        return B

    """ ============== drawing ================ """

    def draw(self, displacements=np.asarray([]), hold=False):
        # draw structure
        if displacements.size==0:
            displacements=np.zeros(2*len(self.nodes))

        for node in self.nodes:
            disp = np.array([ displacements[node.index*2] , displacements[node.index*2+1] ])
            node.disp_pos = node.pos + disp

        fig=plt.figure(figsize=(8,8))
        ax = plt.subplot(121)
        x=[node.disp_pos[0] for node in self.nodes]
        y=[node.disp_pos[1] for node in self.nodes]

        indices=[node.index for node in self.nodes]
        plt.scatter(x,y)
        for i, txt in enumerate(indices):
            plt.annotate(txt, (x[i]+0.1, y[i]-0.05))

        for i, member in enumerate(self.members):
            a=member.nodeA.disp_pos
            b=member.nodeB.disp_pos
            X=np.asarray([a,b]).T
            lab='member ' + str(i)
            plt.plot(*X, label=lab)

        # rescale plot
        plt.axis('square')
        xmin, xmax, ymin, ymax = plt.axis()
        plt.xlim((xmin-0.5, xmax+0.5))
        plt.ylim((ymin-0.5, ymax+0.5))
        plt.legend(bbox_to_anchor= (1.02, 0., 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)
        if not hold:
            plt.show()

    def plot_vector(self, vector):
        self.draw(hold=True)
        XY=np.asarray([node.pos for node in self.nodes])
        UV=np.reshape(vector, np.array([vector.shape[0]/2,2]).astype(int))

        plt.quiver(XY[:,0],XY[:,1], UV[:,0], UV[:,1], scale=1, scale_units='xy')
        plt.show()


class Model:
    def __init__(self, geom):
        # get geometry and make a list of the B's
        self.geom = geom
        # self.zero_disp_constraints = zero_disp_constraints
        # self.remove_indices = [ constr[0]*2+constr[1] for constr in self.zero_disp_constraints ]

        # Bs and Ks will be kxk with k 2n-len(zero_disp_constraints)
        self.Bs=[]
        self.n=len(geom.nodes)
        self.k=2*self.n # -len(zero_disp_constraints)
        self.m=geom.m

        for member in self.geom.members:
            B = geom.mapB(member)
            self.Bs.append(B)

    def makeK(self,b):
        assert(len(b)==self.m)
        K=np.zeros((self.k,self.k))
        for i, Bi in enumerate(self.Bs):
            K+=b[i]*Bi
        return K

    def solve_for(self,b,f,d):
        # if a d is set to not None, the corresponding f doesnt have any effect.
        # deal with displacement BCs
        rows_to_go=[]
        rows_to_stay=[]
        dd=d.copy()
        for i, val in enumerate(d):
            if val is not None:
                rows_to_go.append(i)
            else:
                rows_to_stay.append(i)
                dd[i]=0

        # displacement BCs
        K=self.makeK(b)
        fprime=f-K.dot(dd)

        Kprime=np.delete(K, rows_to_go,0)
        Kprime=np.delete(Kprime, rows_to_go,1)
        fprime=np.delete(fprime, rows_to_go)
        dprime = list(np.linalg.solve(Kprime, fprime))

        dout=dd.copy()
        for i in range(len(dprime)):
            dout[rows_to_stay[i]]=dprime[i]
        dout=np.array(dout)

        return K.dot(dout), dout

    def drawdisp(self, d):
        self.geom.draw(d)
