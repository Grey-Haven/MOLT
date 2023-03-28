# Plotting Utilities
import numpy as np
import matplotlib.pyplot as plt


def make_mesh_plot(field, x, y, field_name, savefig = False):
    """Makes a mesh plot with colorbars of a field quantity."""
    
    if not isinstance(field_name, str):
        
        field_name = str(field_name)
    
    plt.figure(figsize=(10,8))

    X, Y = np.meshgrid(x, y, indexing='ij')

    plot = plt.pcolormesh(X, Y, field, cmap = 'viridis', shading='auto')
    plt.title(field_name, fontsize=32)
    plt.xlabel("$x$",fontsize=32)
    plt.ylabel("$y$",fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    
    ax = plt.gca()
    ax.xaxis.offsetText.set_fontsize(32)
    ax.yaxis.offsetText.set_fontsize(32)
    
    # Use 5 ticks on each side
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    cbar = plt.colorbar(plot)
    cbar.ax.tick_params(labelsize=32)
    cbar.ax.yaxis.offsetText.set(size=32)
    
    if savefig == True:
        
        plt.savefig(field_name + ".png")
        
    else:
        
        plt.show()
    
    return

def make_cont_plot(field, x, y, field_name, savefig = False):
    """Makes a contour plot with colorbars of a field quantity."""
    
    if not isinstance(field_name, str):
        
        field_name = str(field_name)
    
    plt.figure(figsize=(10,8))

    X, Y = np.meshgrid(x, y, indexing='ij')

    plot = plt.contourf(X, Y, field, cmap = 'viridis')
    plt.title(field_name, fontsize=32)
    plt.xlabel("$x$",fontsize=32)
    plt.ylabel("$y$",fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    
    ax = plt.gca()
    ax.xaxis.offsetText.set_fontsize(32)
    ax.yaxis.offsetText.set_fontsize(32)
    
    # Use 5 ticks on each side
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    cbar = plt.colorbar(plot)
    cbar.ax.tick_params(labelsize=32)
    cbar.ax.yaxis.offsetText.set(size=32)
    
    if savefig == True:
        
        plt.savefig(field_name + ".png")
        
    else:
        
        plt.show()
    
    return

### Do we need this???
def plot_elec_slice(x2_elec, p2_elec, x2_grid, slice_locs, savefig = False):
    """Plot the electrons in phase space along a fixed slice (x1 = 0)."""

    plt.figure(figsize=(8,6))
    plt.plot(x2_elec[slice_locs], p2_elec[slice_locs], color = 'blue', 
             lw=0, marker='o', ms=2)

    plt.xlabel(r'$x^{(2)}$', fontsize=20)
    plt.ylabel(r'$p^{(2)}$', fontsize=20)
    plt.title( r"Electron Particle Distribution", fontsize=20)
    plt.grid(linestyle='--')

#     plt.xlim(x2_grid[0], x2_grid[-1])
#     plt.ylim(vmin, vmax)

    if savefig == True:
        
        plt.savefig(field_name + ".png")
        
    else:
        
        plt.show()
    
    return