import matplotlib.pyplot as plt
import numpy as np
from pydicom import dcmread

# A scroll-wheel controlled sliceviewer that views 3d numpy arrays
# taking the 1st dimension of the arrays as the different slices
# retrieved from:
# https://matplotlib.org/3.3.0/gallery/event_handling/image_slices_viewer.html


class IndexTracker:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X

        if len(np.shape(X)) == 4:
            self.slices, rows, cols, chnls = X.shape
        if len(np.shape(X)) == 3:
            self.slices, rows, cols = X.shape

        self.ind = self.slices//2

        if len(np.shape(X)) == 4:
            self.im = ax.imshow(self.X[self.ind, :, :, :])
        if len(np.shape(X)) == 3:
            self.im = ax.imshow(self.X[self.ind, :, :])

        self.update()

    def on_scroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if len(np.shape(self.X)) == 4:
            self.im.set_data(self.X[self.ind, :, :, :])
        if len(np.shape(self.X)) == 3:
            self.im.set_data(self.X[self.ind, :, :])

        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


if __name__ == '__main__':

    plt.gray()

    fig, ax = plt.subplots(1, 1)
    # X is the 3D array of slices we want to examine
    #X = slices
    tracker = IndexTracker(ax, X)

    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()
