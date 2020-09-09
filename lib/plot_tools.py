import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'left':
        move_slice(ax,-1)
    elif event.key == 'right':
        move_slice(ax,1)
    fig.canvas.draw()

def move_slice(ax, shift):
    """Go to the previous slice."""
    ax.clear()
    ax.index = (ax.index + shift) % len(ax.time)  # wrap around using %
    plot_slice(ax) 

def plot_slice(ax):
    for i in range(len(ax.xlist)):
        if ax.ylabel is not None:
            ax.set_ylabel(ax.ylabel)
        if ax.xlabel is not None:
            ax.set_xlabel(ax.xlabel)
        x=ax.xlist[i]
        y=ax.ylist[i]
        if len(y.shape)>len(x.shape):
            x=np.tile(x,(len(y),1))
        if ax.uncertaintylist is not None:
            uncertainty=ax.uncertaintylist[i]
            if uncertainty is not None:
                if ax.labels is not None:
                    ax.errorbar(x[ax.index],
                                y[ax.index],
                                uncertainty[ax.index],
                                label=ax.labels[i],
                                ls='None')
                else:
                    ax.errorbar(x[ax.index],
                                y[ax.index],
                                uncertainty[ax.index],
                                ls='None')
                continue
        if ax.labels is not None:
            ax.plot(x[ax.index],
                    y[ax.index],
                    label=ax.labels[i])
        else:
            ax.plot(x[ax.index],
                    y[ax.index])
    ax.axhline(0,c='k',alpha=.5)
    if ax.ylims is not None:
        ax.set_ylim(ax.ylims)
    if ax.time is not None:
        ax.set_title('{}ms'.format(ax.time[ax.index]))
    if ax.labels is not None:
        ax.legend()
    
               
def plot_comparison_over_time(xlist, ylist, time,
                              ylabel=None,
                              xlabel=None,
                              uncertaintylist=None, 
                              labels=None):
    ymax=max([np.nanmax(y) for y in ylist])
    ymin=min(0,min([np.nanmin(y) for y in ylist]))

    fig, ax = plt.subplots()
    ax.xlist=xlist 
    ax.ylist=ylist
    ax.time=time
    ax.uncertaintylist=uncertaintylist
    ax.ylabel=ylabel
    ax.xlabel=xlabel
    ax.ylims=((ymin,ymax))
    ax.labels=labels
    ax.index=0
    plot_slice(ax)
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def plot_2d_comparison(x_axis, y_axis, data_2d,
                       x_scatter_data, y_scatter_data, scatter_data,
                       xlims=None, ylims=None,
                       xlabel=None, ylabel=None, title=None):
    vmin=min((np.min(data_2d),np.min(scatter_data)))
    vmax=max((np.max(data_2d),np.max(scatter_data)))

    #cs=plt.contourf(x_axis, y_axis, data_2d,
    #                cmap=cm.viridis,vmin=vmin,vmax=vmax)
    #plot a first time so you can see the actual fit before the overlayed data
    plt.show()
    cs=plt.contourf(x_axis, y_axis, data_2d,
                    cmap=cm.viridis,vmin=vmin,vmax=vmax)
    plt.scatter(x_scatter_data, y_scatter_data, 100, scatter_data,
                cmap=cs.cmap, vmin=vmin,vmax=vmax,edgecolors='k',linewidths=.1)
    if title is not None:
        plt.title(title)
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()
    
