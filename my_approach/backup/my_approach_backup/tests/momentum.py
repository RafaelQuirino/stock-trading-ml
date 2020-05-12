import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/rafael/Documents/workspace/github/stock-trading-ml/my_approach')
import util

def import_data(csv_file, column_name, T=0):
    return np.array(pd.read_csv(csv_file)[column_name])[T:]

def normalize_range(vec, a, b):
    vec_norm = vec
    vec_min = np.min(vec)
    vec_max = np.max(vec)
    vec_norm -= vec_min
    vec_norm *= (b-a)
    vec_norm /= (vec_max-vec_min)
    vec_norm += a
    return vec_norm

def get_momentum(vec, winsize=3, mtype=1, matype=1):
    mm = np.zeros(len(vec))
    vec_sma = None
    if matype == 1:
        vec_sma = util.moving_average(vec, winsize=winsize)
    elif matype == 2:
        vec_sma = util.weighted_moving_average(vec, winsize=winsize)
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

if __name__ == '__main__':
    ts = import_data('../data/vale_2019.csv', 'val1', -390*1)
    ts_norm = normalize_range(ts, -1, 1)
    winsize = int(len(ts)/16)

    mm = get_momentum(ts, winsize=winsize, mtype=1, matype=1)
    mm2 = get_momentum(ts, winsize=winsize, mtype=1, matype=2)
    mm3 = get_momentum(ts, winsize=winsize, mtype=2, matype=1)
    mm4 = get_momentum(ts, winsize=winsize, mtype=2, matype=2)
    mm_norm = normalize_range(mm, -1, 1)
    mm_norm2 = normalize_range(mm2, -1, 1)
    mm_norm3 = normalize_range(mm3, -1, 1)
    mm_norm4 = normalize_range(mm4, -1, 1)

    grads = np.gradient(util.moving_average(ts,winsize=winsize)) * (1.0+math.sqrt(5))
    grads_norm = util.moving_average(normalize_range(grads, -1, 1), winsize=winsize)

    fig, ax = plt.subplots()
    util.plot_bollinger_bands(ts_norm,ax)
    ax.plot(ts_norm, alpha=1.0, linewidth=3, label='ts', color='red')
    # ax.plot(grads_norm, alpha=0.7, linewidth=3, label='grads')
    # ax.plot(mm_norm, alpha=0.7, linewidth=2, label='mm normal sma', color='green')
    # ax.plot(mm_norm2, alpha=0.7, linewidth=2, label='mm normal wma', color='blue')
    # ax.plot(mm_norm3, alpha=0.5, linewidth=2, label='mm cumdiff sma')
    # ax.plot(mm_norm4, alpha=0.5, linewidth=2, label='mm cumdiff wma')
    ax.plot(util.weighted_moving_average(ts_norm, winsize=20), label='sma-20', color='magenta')
    ax.plot(util.weighted_moving_average(ts_norm, winsize=50), label='sma-50', color='cyan')
    ax.plot(np.zeros(len(ts)), alpha=0.3, linestyle='dashed')
    plt.legend()
    plt.show()
