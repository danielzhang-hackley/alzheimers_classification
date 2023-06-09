import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import time

matplotlib.use('TkAgg')

 
fig, ax = plt.subplots()


# Adjust the bottom size according to the
# requirement of the user
plt.subplots_adjust(bottom=0.25)



data = pd.read_csv(r'../../data/cnv.csv').drop(columns="Unnamed: 0").filter(regex='HighQual').to_numpy()

t = np.arange(0.0, 100.0, 0.1)
s = np.sin(2*np.pi*t)
 


# plot the x and y using plot function
l = plt.plot(t, s)
 
# Choose the Slider color
slider_color = 'White'
 
# Set the axis and slider position in the plot
axis_position = plt.axes([0.2, 0.1, 0.65, 0.03],
                         facecolor=slider_color)
slider_position = Slider(axis_position,
                         'Pos', 0.1, 90.0)
 
# update() function to change the graph when the
# slider is in use
def update(val):
    pos = slider_position.val
    ax.axis([pos, pos+10, -1, 1])
    fig.canvas.draw_idle()
 
# update function called using on_changed() function
slider_position.on_changed(update)
 
# Display the plot
plt.show()
