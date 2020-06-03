#!/home/rafael/Documents/workspace/github/stock-trading-ml/venv/bin/python3

import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import time_series as ts
import simulation as s
import robot as r

if __name__ == '__main__':

    # prices, datetimes, file = ts.import_data('data/vale3_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/abev3_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/bbas3_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/itub4_2019.csv', 'preco', 'hora', 0)
    prices, datetimes, file = ts.import_data('data/jbss3_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/petr4_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/petr4_2020.csv', 'preco', 'hora', 0)
    sim = s.Simulation(prices, datetimes, file, balance=1000.0)

    T, ndays, lag = 420, 1, 1
    if lag > 0:
        days = prices[-T*ndays*(lag+1):-T*ndays*(lag)]
    else:
        days = prices[-T*ndays:]

    sma_flag = True
    ema_flag = True
    bb_flag  = True
    mo_flag  = True
    rsi_flag = True
    cci_flag = True
    so_flag  = True

    # Simple Moving Average -----------------------------------------------
    if sma_flag:
        sgn = days
        sma = ts.SMA(sgn, winsize=20)
        plt.title(label='Simple Moving Average')
        plt.plot(sgn, label='sgn')
        plt.plot(sma, label='sma')
        plt.legend()
        plt.show()
    #---------------------------------------------------------------------------

    # Exponential Moving Average -----------------------------------------------
    if ema_flag:
        sgn = days
        ema = ts.EMA(sgn)
        plt.title(label='Exponential Moving Average')
        plt.plot(sgn, label='sgn')
        plt.plot(ema, label='ema')
        plt.legend()
        plt.show()
    #---------------------------------------------------------------------------

    # Bollinger Bands ----------------------------------------------------------
    if bb_flag:
        sgn = days
        ma, loband, hiband = ts.BB(sgn)
        fig, ax = plt.subplots()
        fig.suptitle('Bollinger Bands', fontsize=16)
        ax.plot(sgn)
        ax.plot(ma, color='red', alpha=0.5, linewidth=1)
        ax.plot(loband, color='green', alpha=0.7, linewidth=1)
        ax.plot(hiband, color='green', alpha=0.7, linewidth=1)
        ax.fill_between(np.arange(0,len(sgn)),loband, hiband, color='green', alpha=0.1)
        plt.show()
    # --------------------------------------------------------------------------

    # Momentum Oscilator -------------------------------------------------------
    if mo_flag:
        sgn = days
        mo = ts.MO(sgn)
        fig, axs = plt.subplots(2)
        fig.suptitle('Momentum Oscilator', fontsize=16)
        axs[0].plot(days, label='prices')
        axs[0].legend()
        axs[1].plot(mo, label='MO')
        axs[1].plot(np.arange(0,len(mo)), np.full(len(mo),100.0), label='100 limit', color='gray', linestyle='dashed')
        axs[1].legend()
        plt.show()
    # --------------------------------------------------------------------------

    # RSI indicator ------------------------------------------------------------
    if rsi_flag:
        lookback_size = 14
        hist = prices[-T*ndays-lookback_size:-T*ndays]
        sgn = np.concatenate((hist,days))
        rsi, lo, hi = ts.RSI(sgn,initsize=lookback_size)
        fig, axs = plt.subplots(2)
        fig.suptitle('Relative Strength Index', fontsize=16)
        axs[0].plot(days, label='prices')
        axs[0].legend()
        axs[1].plot(rsi, label=f'RSI lookback({lookback_size})')
        axs[1].plot(np.arange(0,len(rsi)), np.full(len(rsi),hi), label=f'{hi}', color='gray', linestyle='dashed')
        axs[1].plot(np.arange(0,len(rsi)), np.full(len(rsi),lo), label=f'{lo}', color='gray', linestyle='dashed')
        axs[1].legend()
        plt.show()
    #---------------------------------------------------------------------------

    # Commodity Channel Index (CCI) --------------------------------------------
    if cci_flag:
        sgn = days
        cci = ts.CCI(sgn)
        fig, axs = plt.subplots(2)
        fig.suptitle('Commodity Channel Index (adapted)', fontsize=16)
        axs[0].plot(days, label='prices')
        axs[0].legend()
        axs[1].plot(ts.SMA(cci), label='CCI')
        axs[1].plot(np.arange(0,len(cci)), np.full(len(cci),100.0),  label=' 100', color='gray', linestyle='dashed')
        axs[1].plot(np.arange(0,len(cci)), np.full(len(cci),-100.0), label='-100', color='gray', linestyle='dashed')
        axs[1].legend()
        plt.show()
    #---------------------------------------------------------------------------

    # Stochastic Oscilator (SO) ------------------------------------------------
    if so_flag:
        sgn = days
        so, lo, hi = ts.SO(sgn)
        fig, axs = plt.subplots(2)
        fig.suptitle('Stochastic Oscilator', fontsize=16)
        axs[0].plot(days, label='prices')
        axs[0].legend()
        axs[1].plot(ts.SMA(so), label='SO')
        axs[1].plot(np.arange(0,len(so)), np.full(len(so),hi), label=f'{hi}', color='gray', linestyle='dashed')
        axs[1].plot(np.arange(0,len(so)), np.full(len(so),lo), label=f'{lo}', color='gray', linestyle='dashed')
        axs[1].legend()
        plt.show()
    #---------------------------------------------------------------------------
