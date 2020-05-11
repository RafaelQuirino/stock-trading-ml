import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
'''
def import_data(csv_file, column_name, date_column_name, T=0):
    df     = pd.read_csv(csv_file)
    prices = np.array(df[column_name])[T:]
    dates  = np.array(df[date_column_name])[T:]
    return prices, dates, csv_file

'''
'''
def normalize_range(vec, a, b):
    vec_norm = vec
    vec_min = np.min(vec)
    vec_max = np.max(vec)
    vec_norm -= vec_min
    vec_norm *= (b-a)
    vec_norm /= (vec_max-vec_min)
    vec_norm += a
    return vec_norm

'''
'''
def get_momentum(vec, winsize=3, mtype=1, matype=1):
    mm = np.zeros(len(vec))
    vec_sma = None
    if matype == 1:
        vec_sma = moving_average(vec, winsize=winsize)
    elif matype == 2:
        vec_sma = weighted_moving_average(vec, winsize=winsize)
    for i in range(len(vec)):
        a, b = i-winsize+1, i+1
        if i < winsize - 1:
            a, b = 0, i+1

        if b-a == 1:
            mm[i] = 0
        else:
            if mtype == 1:
                mean_diff = 0.0
                for i in range(a+1,b):
                    mean_diff += (vec_sma[i] - vec_sma[i-1])
                mm[i] = mean_diff / float(b-(a+1))
            elif mtype == 2:
                mm[i] = vec_sma[b-1]-vec_sma[a] #/ float(winsize)
    return mm

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
def moving_average(sgn, winsize=3, mtype='left', start=0, end=-1):
    if end == -1:
        end = len(sgn)

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
'''
def weighted_moving_average(sgn, winsize=3):
    wma = np.zeros(len(sgn))
    for i in range(len(sgn)):
        a, b = i-winsize+1, i+1
        if i < winsize - 1:
            a, b = 0, i+1

        n = b-a
        w = np.arange(1,n+1)
        v = w * sgn[a:b]
        wma[i] = v.sum() / w.sum()

    return wma

'''
'''
def exponential_moving_average(sgn, winsize=3, smoothing=4.0):
    ema = np.zeros(len(sgn))
    for i in range(len(sgn)):
        a, b = i-winsize+1, i+1
        if i < winsize - 1:
            a, b = 0, i+1

        n = b-a
        alpha = smoothing/(n+1)
        w = np.arange(0,n)[::-1]
        w = np.apply_along_axis(lambda x: (1-alpha)**x, 0, w)

        v = w * sgn[a:b]
        ema[i] = v.sum() / w.sum()

    return ema

'''
TODO
'''
def moving_medians(sgn, winsize=3):
    return []

'''
'''
def bollinger_bands(sgn, winsize=20, stdmult=2.0, autoadjust=False):
    if autoadjust:
        winsize = int(len(sgn) / 10.0)
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
def plot_bollinger_bands(sgn, winsize=20, stdmult=2.0, autoadjust=False, ax = None):
    if autoadjust:
        winsize = int(len(sgn) / 10.0)
    ma, loband, hiband = bollinger_bands(sgn, winsize, stdmult)

    if ax == None:
        fig, ax = plt.subplots()
    # ax.plot(sgn)
    ax.plot(ma, color='red', alpha=0.5, linewidth=1, linestyle='dashed')
    ax.plot(loband, color='green', alpha=0.5, linewidth=1)
    ax.plot(hiband, color='green', alpha=0.5, linewidth=1)
    ax.fill_between(np.arange(0,len(sgn)),loband, hiband, color='green', alpha=0.1)
    # zp = util.ZoomPan(ax, base_scale=1.1)
    # plt.show()
    # zp.destroy()
    return None

def show_bollinger_bands(sgn, winsize=20, stdmult=2.0, autoadjust=False):
    if autoadjust:
        winsize = int(len(sgn) / 10.0)
    ma, loband, hiband = bollinger_bands(sgn, winsize, stdmult)

    fig, ax = plt.subplots()
    ax.plot(sgn)
    ax.plot(ma, color='red', alpha=0.5, linewidth=1)
    ax.plot(loband, color='green', alpha=0.7, linewidth=1)
    ax.plot(hiband, color='green', alpha=0.7, linewidth=1)
    ax.fill_between(np.arange(0,len(sgn)),loband, hiband, color='green', alpha=0.1)
    zp = util.ZoomPan(ax, base_scale=1.1)
    plt.show()
    zp.destroy()
    return None

'''
'''
def plot(sgn):
    fig, ax = plt.subplots()
    ax.plot(sgn)
    zp = ZoomPan(ax, base_scale=1.1)
    plt.show()
    zp.destroy()
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
