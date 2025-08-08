# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time



def state_update(*args):
    """
    Update the state of the Game of Life and the image.
    """
    
    global state_current, image_data

    ###########################################################################
    # Count the number of neighbours
    #
    state_neighbors = ( state_current[0:-2, 0:-2] + state_current[0:-2, 1:-1] + state_current[0:-2, 2:] +
                        state_current[1:-1, 0:-2] + 0.0                       + state_current[1:-1, 2:] +
                        state_current[2:  , 0:-2] + state_current[2:  , 1:-1] + state_current[2:  , 2:])
    
    ###########################################################################
    # recompute current state 

    # Compute all the cell births
    birth = (state_neighbors == 3) & (state_current[1:-1, 1:-1] == 0)
    # Compute all the cell survivals
    survive = ((state_neighbors == 2) | (state_neighbors == 3)) & (state_current[1:-1, 1:-1] == 1)
    # reset the current state to zero
    state_current[...] = 0
    # set the cell as alive if it is born or survives
    state_current[1:-1, 1:-1][birth | survive] = 1

    ###########################################################################
    # Update the image
    #

    show_past_enabled = True
    if show_past_enabled:
        # Never more than 25% from the past - thus current activity is clearly visible
        image_data[image_data>0.25] = 0.25
        # Decrease past activities
        image_data *= 0.98
        # Update with current activity
        image_data[state_current==1] = 1
    else:
        # Direct activity
        image_data[...] = state_current

    # Update the image
    axes_image.set_data(image_data)

state_update.call_count = 0
state_update.time_start = None


# Initialize the state of the Game of Life
state_current = np.random.randint(0, 2, (1000, 1000), dtype=np.int64)

# declare image array
image_data = np.zeros(state_current.shape)

dpi = 80.0
figure_size = state_current.shape[1]/float(dpi), state_current.shape[0]/float(dpi)
figure = plt.figure(figsize=figure_size, dpi=dpi)
figure.add_axes([0.0, 0.0, 1.0, 1.0])
plt.axis('off')
axes_image = plt.imshow(image_data, interpolation='nearest', cmap=plt.cm.gray_r, vmin=0, vmax=1)

animation = FuncAnimation(figure, state_update, interval=0, frames=2000)
# animation.save('game-of-life.mp4', fps=40, dpi=80, bitrate=-1, codec="libx264",
#                extra_args=['-pix_fmt', 'yuv420p'],
#                metadata={'artist':'Nicolas P. Rougier'})
plt.show()