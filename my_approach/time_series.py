#!/home/rafael/Documents/workspace/github/stock-trading-ml/venv/bin/python3

import util
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
'''
def import_data(csv_file, price_column_name, datetime_column_name, T=0):
    df         = pd.read_csv(csv_file)
    prices     = np.array(df[price_column_name])[T:]
    datetimes  = np.array(df[datetime_column_name])[T:]
    return prices, datetimes, csv_file

'''
'''
def normalize_range(sgn, a, b, start=0, end=-1):
    sgn_norm = sgn
    sgn_min = np.min(sgn)
    sgn_max = np.max(sgn)
    sgn_norm -= sgn_min
    sgn_norm *= (b-a)
    sgn_norm /= (sgn_max-sgn_min)
    sgn_norm += a
    return sgn_norm

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
def moving_average(sgn, winsize=3, mtype='left', start=0, end=-1):
    if end == -1:
        end = len(sgn)

    ma = np.zeros(len(sgn))
    for i in range(len(sgn)):
        a, b = i-winsize+1, i+1
        if i < winsize - 1:
            a, b = 0, i+1

        winsum = np.sum(sgn[a:b])
        ma[i] = winsum / float(b-a)

    return ma

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
TODO
'''
def moving_medians(sgn, winsize=3):
    return []

'''
'''
def sample_mean(sgn):
    return sgn.sum() / len(sgn)

'''
'''
def sample_variance(sgn):
    return (np.square(sgn-sample_mean(sgn)).sum()) / len(sgn)

'''
'''
def autocovariance(sgn):
    return None

'''
TODO
'''
def autocorrelation_function(sgn):
    return []

#-------------------------------------------------------------------------------
# Specific trading indicators
#-------------------------------------------------------------------------------
'''
Simple Moving Average (SMA)
'''
def SMA (sgn, winsize=20):
    if winsize > len(sgn):
        winsize = len(sgn)
    return moving_average(sgn, winsize=winsize)

'''
Exponential Moving Average (EMA)
'''
def EMA (sgn):
    ema = np.zeros(len(sgn))
    ema[0] = sgn[0]
    n = len(sgn)
    k = 2.0 / float(n+1)
    for i in range(1,n):
        ema[i] = sgn[i]*k + ema[i-1]*(1.0-k)
    return ema

'''
Bollinger Bands (BB)
'''
def BB (sgn, winsize=20, stdmult=2.0, autoadjust=False):
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
Momentum Oscilator (MO)
'''
def MO (sgn, winsize=20, limit=100.0):
    mo = np.zeros(len(sgn))
    for i in range(len(mo)):
        a, b = i-winsize+1, i+1
        if i < winsize - 1:
            a, b = 0, i+1
        mo[i] = (sgn[i] / sgn[a]) * limit
    return mo, limit

'''
Relative Strength Index (RSI)
=> Returns an array of size len(sgn)-initsize
'''
def RSI(sgn, lothreshold=30.0, hithreshold=70.0, initsize=14): # Standard literature values
    rsi = np.zeros(len(sgn))

    upsum,   upcount   = 0.0, 0
    downsum, downcount = 0.0, 0

    for i in range(1,initsize):
        base = sgn[i-1]
        if sgn[i] > sgn[i-1]:
            diff = sgn[i] - sgn[i-1]
            upsum   += (diff*100.0)/base
            upcount += 1
        elif sgn[i-1] > sgn[i]:
            diff = sgn[i-1] - sgn[i]
            downsum   += (diff*100.0)/base
            downcount += 1

    average_gain = upsum / float(upcount)     if upcount > 0   else upsum / 1.0
    average_loss = downsum / float(downcount) if downcount > 0 else downsum / 1.0
    if average_loss == 0:
        average_loss = 1.0 / float(initsize)

    scale = 100.0

    rsi[0] = scale - (scale / (1 + (average_gain/average_loss)))

    for i in range(initsize,len(sgn)):
        current_gain = 0.0 if sgn[i] <= sgn[i-1] else ((sgn[i]-sgn[i-1])*100.0)/sgn[i-1]
        current_loss = 0.0 if sgn[i-1] <= sgn[i] else ((sgn[i-1]-sgn[i])*100.0)/sgn[i-1]
        num = (average_gain * (initsize-1) + current_gain) / float(initsize)
        den = (average_loss * (initsize-1) + current_loss) / float(initsize)

        rsi[i-initsize-1] = scale - (scale / (1 + (num/den)))

        average_gain = num
        average_loss = den

    return rsi, lothreshold, hithreshold

'''
Commodity Channel Index (CCI)
Obs.: This indicator was adapted to daytrading minute-to-minute day_prices,
      i.e., no high/low/close observations.
'''
def CCI (sgn, winsize=20):
    cci = np.zeros(len(sgn))
    for i in range(len(cci)):
        a, b = i-winsize+1, i+1
        if i < winsize - 1:
            a, b = 0, i+1
        k = 0.015   # Literature constant
        tp = sgn[i] # Typical price here is just the price
        ma = np.sum(sgn[a:b]) / float(b-a)
        md = np.sum(np.abs(sgn[a:b] - ma)) / float(b-a)
        if md == 0:
            md = 1e-9
        cci[i] = (tp - ma) / (k * md)
    return cci

'''
Stochastic Oscilator (SO)
'''
def SO (sgn, lothreshold=20, hithreshold=80, winsize=14): # Standard literature values
    so = np.zeros(len(sgn))
    for i in range(len(so)):
        a, b = i-winsize+1, i+1
        if i < winsize - 1:
            a, b = 0, i+1
        lo = np.min(sgn[a:b])
        hi = np.max(sgn[a:b])
        if hi - lo == 0:
            hi += 1
        so[i] = ((sgn[i] - lo) / (hi - lo)) * 100.0
    return so, lothreshold, hithreshold
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Invented day trading indicators
#-------------------------------------------------------------------------------
def BB_PEAKS(sgn, bb):
    if len(sgn) != len(bb[0]):
        return None

    bb_peaks = np.zeros(len(sgn))

    IDLE     = 0
    ENTER_LO = 1
    ENTER_HI = 2
    state    = IDLE
    pos      = None
    for i in range(len(sgn)):
        if sgn[i] <= bb[1][i]:
            state = ENTER_LO
            pos = i
        elif sgn[i] >= bb[2][i]:
            state = ENTER_HI
            pos = i
        else:
            state = IDLE

        if state == ENTER_LO:
            area = 0.0
            for k in range(pos,i):
                area += sgn[k] - bb[1][k]
            area /= ((i-pos)+1)
            bb_peaks[i] = area
        elif state == ENTER_HI:
            area = 0.0
            for k in range(pos,i):
                area += sgn[k] - bb[2][k]
            area /= ((i-pos)+1)
            bb_peaks[i] = area

    plt.plot(bb_peaks)
    plt.show()
    # exit()

    return bb_peaks

def BB_PEAKS_2 (sgn, bb):
    if len(sgn) != len(bb[0]):
        return None

    bb_peaks = np.zeros(len(sgn))

    for i in range(len(sgn)):
        p = sgn[i]
        if p < bb[1][i]:
            bb_peaks[i] = p - bb[1][i]
        elif p > bb[2][i]:
            bb_peaks[i] = p - bb[2][i]

    return bb_peaks
#-------------------------------------------------------------------------------
