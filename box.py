import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable

import h5py as h5
import sys, os
import matplotlib.ticker as ticker




class Box:

    #location of (0,0,0) corner
    x0 = 0.0
    y0 = 0.0
    z0 = 0.0

    # box size
    dx = 1.0
    dy = dx
    dz = dy

    #default plotting style
    #fmt={'color':'k','linestyle':'solid',}


    #ctor
    def __init__(self, ax_in):
        self.ax = ax_in


    # initialize box corners
    def make_corners(self):
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0

        dx = self.dx
        dy = self.dy
        dz = self.dz
        
        #xyz corners
        self.corners = np.array(list(product([x0, x0+dx], [y0, y0+dy], [z0, z0+dz]))) 

        #print("corners are:")
        #print(self.corners)

    # filter given points away from corner array to get visibility conditions
    def filter_points(self, pps):
        corners = []

        #print("filttering...")
        self.make_corners() #update corners; just in case
        for pp in self.corners:
            #print(" drawing ", pp)

            flag=True #there is no such a point
            for ps in pps:
                if (pp == ps).all():
                    flag = False

            if flag:
                #print("     accepted")
                corners.append( pp )
        return corners


    def make_panel(self, corners):
        panel = []
        
        #draw cube
        #print("combinatorics for:", corners)
        for s, e in combinations( corners, 2):
            if np.sum(np.abs(s-e)) == self.dx:
                #print("XX    ", s,e )
                panel.append( (s,e) )

        return panel


    # initialize box outlines
    def draw_outline(self,
            fmt={'color':'k','linestyle':'solid',}
            ):
        self.make_corners()

        #outlines = self.make_panel(self.corners)
        #for (p0, p1) in outlines:
        #    #print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
        #    lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
        #    lines.set(**self.fmt)
        #print("-------------filttering...")

        corners_f = self.filter_points( [ np.array([self.x0, self.y0, self.z0]) ] )
        outlines2 = self.make_panel( corners_f )

        #print(outlines2)
        for (p0, p1) in outlines2:
            #print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
            lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
            lines.set(**fmt)


    # draw also backside and bottom panels using exploded view
    def draw_hidden_panels(self, 
            side, 
            off = dx,
            fmt={'color':'k','linestyle':'dashed',},
            ):

        farr = []
        if side == "bottom":
            for pp in self.corners:
                #print("drawing", pp, pp[2])
                if not(pp[2] == self.z0):
                    #print("appending")
                    farr.append(pp)

            cors = self.filter_points( farr )

            #add offset
            cors2 = []
            for i in range(len(cors)):
                cors2.append( cors[i] + np.array([0,0,-off]) )

            #print("bottom panel=", cors2)
            outlines = self.make_panel( cors2 )

        elif side == "right":
            for pp in self.corners:
                #print("drawing", pp, pp[2])
                if not(pp[0] == self.x0):
                    #print("appending")
                    farr.append(pp)

            cors = self.filter_points( farr )

            #add offset
            cors2 = []
            for i in range(len(cors)):
                cors2.append( cors[i] + np.array([-off,0,0]) )

            #print("right panel=", cors2)
            outlines = self.make_panel( cors2 )

        elif side == "left":
            for pp in self.corners:
                #print("drawing", pp, pp[2])
                if not(pp[1] == self.y0):
                    #print("appending")
                    farr.append(pp)

            cors = self.filter_points( farr )

            #add offset
            cors2 = []
            for i in range(len(cors)):
                cors2.append( cors[i] + np.array([0,-off,0]) )

            #print("left panel=", cors2)
            outlines = self.make_panel( cors2 )


        #print(outlines)
        for (p0, p1) in outlines:
            #print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
            lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
            lines.set(**fmt)




def draw_box_surface(
        ax, 
        kname, 
        zoff, 
        cmap=plt.cm.RdBu, 
        vmin=0.0, 
        vmax=1.0,
        ):

    data  = combine_tiles(files_F[0], kname, conf)[:,:,0]
    
    
    ny, nx = np.shape(data)
    print("nx={} ny={}".format(nx, ny))
    
    X, Y = np.meshgrid( np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
    Z = zoff*np.ones(X.shape)
    
    #np.clip(data, vmin, vmax, out=data)
    #data = data/data.max()

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    ax.plot_surface(
            X,Y,Z,
            rstride=5,
            cstride=5,
            facecolors=cmap( norm( data ) ),
            shade=False,
            alpha=1.0,
            antialiased=True,
            )

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


##################################################
# plot
if __name__ == "__main__":

    
    fig = plt.figure(figsize=(3.54, 3.0)) #single column fig
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    
    plt.rc('font', family='serif', size=7)
    plt.rc('xtick')
    plt.rc('ytick')
    
    axs = []
    axs.append( fig.add_subplot(111, projection='3d') )
    
    
    box = Box(axs[0])
    box.draw_outline()
    box.draw_hidden_panels("bottom")
    box.draw_hidden_panels("left")
    box.draw_hidden_panels("right")
    

    axs[0].set_axis_off()
    axs[0].view_init(30.0, 45.0)
    
    

    axisEqual3D(axs[0])
    fname = 'box'
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.1, top=1.0)
    plt.savefig(fname+'.pdf')
    plt.savefig(fname+'.png')
    
