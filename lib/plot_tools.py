import matplotlib.pyplot as plt
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
    ax.index = (ax.index + shift) % ax.signal.shape[0]  # wrap around using %
    plot_slice(ax) #ax.images[0].set_array(volume[ax.index])

def plot_slice(ax):
    max_uncertainty=ax.max_uncertainty
    ax.plot(ax.standard_psi, ax.signal[ax.index], c='r')
    ax.errorbar(ax.psi[:,ax.index],
                ax.value[:,ax.index],
                np.clip(ax.uncertainty[:,ax.index],0,max_uncertainty),
                ls='None')
    ax.set_xlabel('psi')
    ax.set_ylabel(ax.final_sig_name)
    ax.set_ylim((0,8))
    ax.set_title('{}ms'.format(ax.standard_times[ax.index]))
    
               
def plot_fit(signal,standard_psi,psi,value,uncertainty,standard_times,final_sig_name,max_uncertainty):
    fig, ax = plt.subplots()
    ax.final_sig_name=final_sig_name
    ax.index=0
    ax.signal=signal
    ax.standard_times=standard_times
    ax.standard_psi=standard_psi
    ax.psi=psi
    ax.value=value
    ax.uncertainty=uncertainty
    ax.max_uncertainty=max_uncertainty
    plot_slice(ax)
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

#plot_fit('temp',np.array([1,1.5,2]),np.array([1,2,3]),np.array([[1,2,3],[2,3,4],[3,4,5]]),np.array([[1,2,3],[2,3,4],[3,4,5]])*.1)

# fig, ax = plt.subplots()

# def on_key(event):
#     key = event.key
#     if key =="left":
#         print('hi')
#         ax.plot([1,2,3])
#         plt.show()
#     elif key == 'right':
#         ax.plot([1,1,1])
            
#     print('you pressed', event.key)

# cid = fig.canvas.mpl_connect('key_press_event', on_key)
# plt.show()
# fig.canvas.mpl_disconnect(cid)
