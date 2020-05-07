import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

'''
THIS CLASS ACTUALLY WORKS !
A class to make matplotlib graphs zoomable and panable with the mouse
'''
from matplotlib.pyplot import figure, show
class ZoomPan:
    def __init__(self, ax, base_scale=1.2):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None
        self.figZoom = self.zoom_factory(ax, base_scale=base_scale)
        self.figPan = self.pan_factory(ax)

    def destroy(self):
        self = None

    def zoom_factory(self, ax, base_scale = 1.2):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        return onMotion

'''
'''
def load_col_from_csv(csvpath, colname):
    df = pd.read_csv(csvpath)
    return np.array(df[colname])

'''
'''
def minmax_normalization(sgn):
    sgn_normalizer = preprocessing.MinMaxScaler()
    sgn_normalized = data_normaliser.fit_transform(sgn)
    return sgn_normalized, normalizer

'''
'''
def minmax_denormalization(sgn_normalized, normalizer):
    sgn_denormalized = normalizer.inverse_transform(sgn_normalized)
    return sgn_denormalized

'''
'''
def moving_average_left(sgn, winsize):
    ma = np.zeros(len(sgn))
    for i in range(len(sgn)):
        a, b = i-winsize+1, i+1
        if i < winsize - 1:
            a, b = 0, i+1
        winsum = sgn[a:b].sum()
        ma[i] = winsum / float(b-a)
    return ma

'''
TODO
'''
def moving_average_center(sgn, winsize):
    return []

'''
TODO
'''
def moving_average_right(sgn, winsize):
    return []

'''
'''
def moving_average(sgn, winsize=3, mtype='left'):
    ma = []
    if mtype == 'left':
        ma = moving_average_left(sgn, winsize)
    elif mtype == 'right':
        ma = moving_average_right(sgn, winsize)
    elif mtype == 'center':
        ma = moving_average_center(sgn, winsize)
    return ma

'''
TODO
'''
def hanning_filter(sgn, winsize=3):
    return []

'''
TODO
'''
def moving_medians(sgn, winsize=3):
    return []

'''
'''
def bollinger_bands(sgn, winsize=20, stdmult=2.0):
    ma = moving_average(sgn,winsize)
    stddevs = np.zeros(len(ma))
    for i in range(len(ma)):
        a, b = i-winsize+1, i+1
        if i < winsize - 1:
            a, b = 0, i+1
        stddevs[i] = np.std(ma[a:b])
    return ma, ma-(stddevs*stdmult), ma+(stddevs*stdmult)

'''
'''
def plot_bollinger_bands(sgn, winsize=20, stdmult=2.0, autoadjust=False):
    if autoadjust:
        winsize = int(len(sgn) / 10.0)
    ma, loband, hiband = bollinger_bands(sgn, winsize, stdmult)

    fig, ax = plt.subplots()
    ax.plot(sgn)
    ax.plot(ma, color='red', alpha=0.5, linewidth=1)
    ax.plot(loband, color='green', alpha=0.7, linewidth=1)
    ax.plot(hiband, color='green', alpha=0.7, linewidth=1)
    ax.fill_between(np.arange(0,len(sgn)),loband, hiband, color='green', alpha=0.1)
    zp = ZoomPan(ax, base_scale=1.1)
    plt.show()
    zp.destroy()
    return None

'''
'''
def plot(sgn):
    plt.plot(sgn)
    plt.show()
    return None

'''
'''
def sample_mean(sgn):
    return sgn.sum() / len(sgn)

'''
'''
def sample_variance(sgn):
    return (np.square(sgn-sample_mean(sgn)).sum()) / len(sgn)

'''
TODO
'''
def autocorrelation_function(sgn):
    return []

'''
=> Main function
'''
if __name__ == '__main__':
    sgn = load_col_from_csv('data/vale_2019.csv','val1')
    T = 390
    sgn = sgn[-T:]

    # plt.plot(sgn)
    # plt.plot(moving_average(sgn,50))
    # plt.show()
    plot_bollinger_bands(sgn, winsize=20, stdmult=2.0, autoadjust=False)

    print(len(sgn))
    print('mean: ', sample_mean(sgn))
    print('variance: ', sample_variance(sgn))
