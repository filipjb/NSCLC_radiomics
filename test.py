import numpy as np
import matplotlib.pyplot as plt
from skimage import color

#%%
#%matplotlib
im = plt.imread(r"C:\Users\filip\OneDrive\Documents\Masteroppgave\Illustrasjoner\grey_image_example.png", )
im = color.rgb2gray(im)

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')

x = y = np.arange(256)
X, Y = np.meshgrid(x, y)

#%%

ax.plot_surface(X, Y, im, cmap="gray")
fig.show()
