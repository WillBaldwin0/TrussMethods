from trusspackage.trusses import Geometry, Model

# some templates structures...

def det_tower(height, width):
    # has a fixed base
    G=Geometry()
    for i in range(height):
        for j in range(width):
            G.addnode([j,i])
            # i,j th node is the (i*width)+j th in list
    for i in range(height-1):
        G.addmember((i+1)*width, i*width)
        for j in range(width-1):
            index = (i+1)*width+1+j
            G.addmember(index, index-1)
            G.addmember(index, index-width)
            G.addmember(index, index-1-width)
    return G

def indet_tower(height, width):
    # has a fixed base
    G=Geometry()
    for i in range(height):
        for j in range(width):
            G.addnode([j,i])
            # i,j th node is the (i*width)+j th in list
    for i in range(height-1):
        G.addmember((i+1)*width, i*width)
        for j in range(width-1):
            index = (i+1)*width+1+j
            G.addmember(index, index-1)
            G.addmember(index, index-width)
            G.addmember(index, index-1-width)
            G.addmember(index-1, index-width)
    return G
