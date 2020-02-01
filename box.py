import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable

import h5py as h5
import sys, os
import matplotlib.ticker as ticker

np.random.seed(0)


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

    def set_data(self, data):
        self.data = data

        self.vmin = np.min( data )
        self.vmax = np.max( data )
        print("vmin = {}".format(self.vmin))
        print("vmax = {}".format(self.vmax))


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
            if np.sum(np.abs(s-e)) == self.dx or \
               np.sum(np.abs(s-e)) == self.dy or \
               np.sum(np.abs(s-e)) == self.dz:
                #print("XX    ", s,e )
                panel.append( (s,e) )

        return panel


    # initialize box outlines
    def draw_outline(self,
            fmt={'color':'k','linestyle':'dashed', 'lw':0.4, 'zorder':20}
            ):
        self.make_corners()

        #outlines = self.make_panel(self.corners)
        #for (p0, p1) in outlines:
        #    #print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
        #    lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
        #    lines.set(**self.fmt)
        #print("-------------filttering...")

        corners_f = self.filter_points( [ np.array([
            self.x0+self.dx, 
            self.y0+self.dy, 
            self.z0]) ] )
        outlines2 = self.make_panel( corners_f )

        #print(outlines2)
        for (p0, p1) in outlines2:
            #print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
            lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
            lines.set(**fmt)


    def _check_for_point(self, arr):
        self.make_corners()
        farr = []
        for pp in self.corners:
            #print("drawing", pp, pp[2])
            for i in [0,1,2]:
                if not(arr[i] == None) and not(pp[i] == arr[i]):
                    #print("appending")
                    farr.append(pp)
        return farr

    def _add_offset(self, cors, offs):
        cors2 = []
        for i in range(len(cors)):
            cors2.append( cors[i] + offs )
        return cors2


    # draw also backside and bottom panels using exploded view
    def draw_exploded_panels_outline(self, 
            side, 
            off = dx,
            fmt={'color':'k','linestyle':'dashed',},
            ):

        if side == "bottom":
            farr = self._check_for_point([None, None, self.z0])
            cors = self.filter_points( farr )
            cors2 = self._add_offset(cors, np.array([0,0,-off]) )
            outlines = self.make_panel( cors2 )

        elif side == "left":
            farr = self._check_for_point([self.x0, None, None])
            cors = self.filter_points( farr )
            cors2 = self._add_offset(cors, np.array([-off,0,0]) )
            outlines = self.make_panel( cors2 )

        elif side == "right":
            farr = self._check_for_point([self.x0+self.dx,None, None])
            cors = self.filter_points( farr )
            cors2 = self._add_offset(cors, np.array([+off,0,0]) )
            outlines = self.make_panel( cors2 )

        elif side == "back":
            farr = self._check_for_point([None, self.y0+self.dy,None])
            cors = self.filter_points( farr )
            cors2 = self._add_offset(cors, np.array([0,+off,0]) )
            outlines = self.make_panel( cors2 )


        #print(outlines)
        for (p0, p1) in outlines:
            #print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
            lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
            lines.set(**fmt)


    def draw_exploded_slice(
            self, 
            side, 
            loc,
            off, 
            cmap='viridis',
            fmt={'color':'k','linestyle':'dotted', 'linewidth':0.5, 'zorder':20},
            fmtC={'color':'k','linestyle':'dotted', 'linewidth':0.3,'zorder':20},
            ):

        if side == "left-front":
            farr = self._check_for_point([self.x0, None, None])
            cors = self.filter_points( farr )
            cors2 = self._add_offset(cors, np.array([loc,-off,0]) )
            outlines = self.make_panel( cors2 )

            #find point in same dim as offset is inserted that is closest to the box
            pp1 = (self.x0 + loc, self.y0, self.z0        )
            pp2 = (self.x0 + loc, self.y0, self.z0+self.dz)
            lines, = self.ax.plot( [pp1[0], pp1[0]], [pp1[1], pp1[1]-off], [pp1[2], pp1[2]], **fmtC)
            lines, = self.ax.plot( [pp2[0], pp2[0]], [pp2[1], pp2[1]-off], [pp2[2], pp2[2]], **fmtC)

            nx,ny,nz = np.shape(self.data)
            iloc = np.int( (loc/self.dx)*nx )
            #print(loc, iloc)
            data_slice = self.data[iloc,:,:]

            ny, nz = np.shape(data_slice)
            Y, Z = np.meshgrid( 
                    np.linspace(self.y0-off, self.y0 + self.dy-off, ny), 
                    np.linspace(self.z0, self.z0 + self.dz, nz))

            x1 = self.x0 + loc
            X = x1*np.ones(Y.shape)


        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)

        #print(outlines)
        for (p0, p1) in outlines:
            #print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
            lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
            lines.set(**fmt)





    def draw_top(self, cmap=plt.cm.viridis):
        farr = self._check_for_point([None, None, self.z0 + self.dz])
        cors = self.filter_points( farr )
        data_slice = self.data[:,:,-1]
    
        nx, ny = np.shape(data_slice)
        X, Y = np.meshgrid( 
                np.linspace(self.x0, self.x0 + self.dx, nx), 
                np.linspace(self.y0, self.y0 + self.dy, ny))
        z1 = self.z0 + self.dz
        Z = z1*np.ones(X.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)

    def draw_left(self, cmap=plt.cm.viridis):
        farr = self._check_for_point([self.x0, None, None])
        cors = self.filter_points( farr )
        data_slice = self.data[0,:,:]

        ny, nz = np.shape(data_slice)
        Y, Z = np.meshgrid( 
                np.linspace(self.y0, self.y0 + self.dy, ny), 
                np.linspace(self.z0, self.z0 + self.dz, nz))
        x1 = self.y0 
        X = x1*np.ones(Y.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)

    def draw_front(self, cmap=plt.cm.viridis):
        farr = self._check_for_point([self.x0, None, None])
        cors = self.filter_points( farr )
        data_slice = self.data[:,0,:]

        nx, nz = np.shape(data_slice)
        X, Z = np.meshgrid( 
                np.linspace(self.x0, self.x0 + self.dx, nx), 
                np.linspace(self.z0, self.z0 + self.dz, nz))
        y1 = self.y0 
        Y = y1*np.ones(X.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)

    def draw_right(self, cmap=plt.cm.viridis):
        farr = self._check_for_point([self.x0 + self.dx, None, None])
        cors = self.filter_points( farr )
        data_slice = self.data[-1,:,:] 

        ny, nx = np.shape(data_slice)
        Y, Z = np.meshgrid( 
                np.linspace(self.y0, self.y0 + self.dy, nx), 
                np.linspace(self.z0, self.z0 + self.dz, ny))
        x1 = self.x0 + self.dx
        X = x1*np.ones(Z.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)


    def draw_exploded_bottom(self, off=dx, cmap=plt.cm.inferno):
        farr = self._check_for_point([None, None, self.z0])
        cors = self.filter_points( farr )
        data_slice = self.data[:,:,0] 

        nx, ny = np.shape(data_slice)
        X, Y = np.meshgrid( 
                np.linspace(self.x0, self.x0 + self.dx, nx), 
                np.linspace(self.y0, self.y0 + self.dy, ny))
        z1 = self.z0 - off
        Z = z1*np.ones(X.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)


    def draw_exploded_back(self, off=dx, cmap=plt.cm.inferno):
        farr = self._check_for_point([self.x0+self.dx, self.y0+self.y0, None])
        cors = self.filter_points( farr )
        data_slice = self.data[:,-1,:] 

        nx, nz = np.shape(data_slice)
        X, Z = np.meshgrid( 
                np.linspace(self.x0, self.x0 + self.dx, nx), 
                np.linspace(self.z0, self.z0 + self.dz, nz))
        y1 = self.y0 + self.dy + off
        Y = y1*np.ones(X.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)

    def draw_exploded_left(self, off=dx, cmap=plt.cm.inferno):
        farr = self._check_for_point([self.x0, None, None])
        cors = self.filter_points( farr )
        data_slice = self.data[0,:,:]

        ny, nz = np.shape(data_slice)
        Y, Z = np.meshgrid( 
                np.linspace(self.y0, self.y0 + self.dy, ny), 
                np.linspace(self.z0, self.z0 + self.dz, nz))

        x1 = self.x0 - off
        X = x1*np.ones(Y.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)


    def draw_exploded_right(self, off=dx, cmap=plt.cm.inferno):
        farr = self._check_for_point([self.x0+self.dx, None, None])
        cors = self.filter_points( farr )
        data_slice = self.data[-1,:,:] 

        ny, nz = np.shape(data_slice)
        Y, Z = np.meshgrid( 
                np.linspace(self.y0, self.y0 + self.dy, ny), 
                np.linspace(self.z0, self.z0 + self.dz, nz))

        x1 = self.x0 + self.dx + off
        X = x1*np.ones(Y.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)


    # mimick volume rendering with multiple opaque slices
    def draw_volume(self):

        nx, ny, nz = np.shape(self.data)
        X, Y = np.meshgrid( 
                np.linspace(self.x0, self.x0 + self.dx, nx), 
                np.linspace(self.y0, self.y0 + self.dy, ny))

        for k in range(ny):
            data_slice = self.data[:,:,k]

            z1 = self.z0 + self.dz*float(k)/float(nz)
            Z = z1*np.ones(X.shape)

            self.draw_surface(X,Y,Z,data_slice, alpha=0.3)




    def draw_surface(self, 
            X, Y, Z, 
            data, 
            cmap = plt.cm.viridis,
            alpha=1.0
            ):

        #print("draw_surface {} {}".format(self.vmin, self.vmax))
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        self.ax.plot_surface(
                X,Y,Z,
                rstride=1,
                cstride=1,
                facecolors=cmap( norm( data.T ) ),
                shade=False,
                alpha=alpha,
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



#for data
def f(x, y, z):
    #u = np.cos(x)*np.sin(y)
    #v = np.sin(x)*np.cos(y)
    #w = np.cos(x)*np.cos(y)
    kx = 1.0
    ky = 5.0
    kz = 10.0
    return np.cos(kx*x + 0.1) + np.cos(ky*y + 0.2) + np.cos(kz*z + 0.3)

#def f(x, y, z):
#    return np.exp(-0.5*(x - 1.0)**2.0) * np.exp(-0.10*(y - 1.0)**2.0) * np.exp(-0.01*(z - 1.0)**2.0)


#def f(x, y, z):
#    sig1 = 1.0
#    sig2 = 2.0
#    sig3 = 3.0
#    norm1 = 1.0/np.sqrt(2.0*np.pi*sig1)
#    norm2 = 1.0/np.sqrt(2.0*np.pi*sig2)
#    norm3 = 1.0/np.sqrt(2.0*np.pi*sig3)
#    return norm1*norm2*norm3*np.exp(-0.5*((x - 1.0)/sig1)**2.0) * np.exp(-0.5*((y - 1.0)/sig2)**2.0) * np.exp(-0.5*((z - 1.0)/sig3)**2.0) 



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
    
    #create data
    Nx = 50
    x = np.linspace(0.0, 1.0, Nx)
    y = np.linspace(0.0, 1.0, Nx)
    z = np.linspace(0.0, 1.0, Nx)
    X, Y, Z = np.meshgrid(x, y, z)
    data = f(X,Y,Z)


    #random data; low-k forcing
    #data = np.zeros((Nx, Nx, np.int(Nx/2)+1 ),dtype=np.complex64)
    #for i in range(1,10):
    #    for j in range(1,10):
    #        for k in range(1,5):
    #            if np.sqrt(i*i + j*j + k*k) > 6:
    #                continue
    #            norm = 1.0/(i + j + k + 1)**2.0
    #            val = np.random.randn() + 1j*np.random.randn()
    #            val /= val*val
    #            data[i,j,k] = norm*val
    #data = np.fft.irfftn(data)
    #print(np.shape(data))

    ################################################## 
    # draw box
    

    box = Box(axs[0])
    box.set_data(data)

    #surface rendering
    box.draw_top()
    box.draw_left()
    box.draw_right()

    #volume rendering
    #box.draw_volume()
    box.draw_outline()


    #back exploded panels
    if True:
        off = 0.7
        box.draw_exploded_panels_outline("bottom", off=off)
        box.draw_exploded_panels_outline("left",   off=off)
        box.draw_exploded_panels_outline("right",  off=off)
        
        box.draw_exploded_bottom(off=off)
        box.draw_exploded_left(  off=off)
        box.draw_exploded_right( off=off)

    if True:
        #front exploded panels
        off = -1.95 #this puts them in front of the box
        cmap = plt.cm.RdBu
        box.draw_exploded_panels_outline("bottom", off=off)
        box.draw_exploded_panels_outline("left",   off=off)
        box.draw_exploded_panels_outline("right",  off=off)
        
        box.draw_exploded_bottom(off=off, cmap=cmap)
        box.draw_exploded_left(  off=off, cmap=cmap)
        box.draw_exploded_right( off=off, cmap=cmap)


    axs[0].set_axis_off()
    axs[0].view_init(35.0, 45.0)


    if False:
        #colorbars
        m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        m.set_array([0.0, 1.0])
        m.set_clim( vmin=0.0, vmax=1.0 )
        cbaxes = fig.add_axes([0.2, 0.91, 0.6, 0.02]) #[left, bottom, width, height],
        cb = plt.colorbar(m, cax = cbaxes, orientation="horizontal", ticklocation="top")  
        fig.text(0.15, 0.91,  r'$n_{\pm}$')

        m = plt.cm.ScalarMappable(cmap=plt.cm.inferno)
        m.set_array([0.0, 1.0])
        m.set_clim( vmin=0.0, vmax=1.0 )
        cbaxes = fig.add_axes([0.2, 0.09, 0.6, 0.02]) #[left, bottom, width, height],
        cb = plt.colorbar(m, cax = cbaxes, orientation="horizontal", ticklocation="top")  
        fig.text(0.15, 0.10,  r'$n_{\nu}$')

        m = plt.cm.ScalarMappable(cmap=plt.cm.RdBu)
        m.set_array([-1.0, 1.0])
        m.set_clim( vmin=-1.0, vmax=1.0 )
        cbaxes = fig.add_axes([0.2, 0.06, 0.6, 0.02]) #[left, bottom, width, height],
        cb = plt.colorbar(m, cax = cbaxes, orientation="horizontal", ticklocation="bottom")  
        #cb.set_label(r'$J$', rotation=0)
        fig.text(0.15, 0.05,  r'$J$')



    axisEqual3D(axs[0])
    fname = 'box'
    plt.subplots_adjust(left=-0.1, bottom=-0.1, right=1.1, top=1.1)
    plt.savefig(fname+'.pdf')
    plt.savefig(fname+'.png')
    
