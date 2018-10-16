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
    fmt={'color':'k','linestyle':'solid',}


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

        print("corners are:")
        print(self.corners)

    # filter given points away from corner array to get visibility conditions
    def filter_points(self, pps):
        corners = []

        self.make_corners() #update corners; just in case
        for pp in self.corners:
            print(" drawing ", pp)
            if (pp == pps).all():
                continue
            print("     accepted")
            corners.append( pp )
        return corners


    def make_panel(self, corners):
        panel = []
        
        #draw cube
        print("combinatorics for:", corners)
        for s, e in combinations( corners, 2):
            if np.sum(np.abs(s-e)) == self.dx:
                print("XX    ", s,e )
                panel.append( (s,e) )

        return panel


    # initialize box outlines
    def make_outline(self):
        self.make_corners()

        outlines = self.make_panel(self.corners)
        for (p0, p1) in outlines:
            print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
            lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
            lines.set(**self.fmt)

        print("-------------filttering...")

        fmt2={'color':'r','linestyle':'dashed',}
        corners_f = self.filter_points( [ np.array([self.x0, self.y0, self.z0]) ] )
        outlines2 = self.make_panel( corners_f )

        print(outlines2)
        for (p0, p1) in outlines2:
            print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
            lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
            lines.set(**fmt2)





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
    box.make_outline()
    
    #axs[0].set_axis_off()
    axs[0].view_init(45.0, 45.0)
    
    
    fname = 'box'
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.1, top=1.0)
    plt.savefig(fname+'.pdf')
    plt.savefig(fname+'.png')
    
