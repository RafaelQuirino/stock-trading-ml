import numpy as np
import simulation as s
import time_series as ts
import matplotlib.pyplot as plt

class Robot:

    ACTION_BUY  = 0
    ACTION_SELL = 1
    ACTION_WAIT = 2

    '''
    '''
    def __init__(self, prices=[], datetimes=[]):
        self.prices    = prices
        self.datetimes = datetimes

    '''
    '''
    def inputPrice (self, newprice, newdatetime):
        self.prices.append(newprice)
        self.datetimes.append(newdatetime)

    '''
    '''
    def forecast (self, winsize=1):
        prediction = []
        for i in range(winsize):
            prediction.append(0.0)
        return prediction

    '''
    '''
    def query (self, newprice=None, newdatetime=None):
        return Robot.ACTION_WAIT

#===============================================================================
# +-------+
# | TESTS |
# +-------+
#
# - Lets try each of the trading indicators
#
#===============================================================================
if __name__ == '__main__':

    prices, datetimes, csvfile = ts.import_data('data/vale_2019.csv', 'val1', 'datetime')
    print(csvfile, '\n')
    print(prices, '\n')
    print(datetimes, '\n')

    sim = s.Simulation(prices, datetimes, csvfile, balance=1000.0)
    print(sim.balance, sim.transaction_tax, '\n')

    T, ndays = 390, 1
    days = prices[-T*ndays:]

    # Simple Moving Average -----------------------------------------------
    sgn = days
    ema = ts.SMA(sgn, winsize=300)
    plt.title(label='Simple Moving Average')
    plt.plot(sgn, label='sgn')
    plt.plot(ema, label='sma')
    # plt.legend()
    # plt.show()
    # exit()
    #---------------------------------------------------------------------------

    # Exponential Moving Average -----------------------------------------------
    sgn = days
    ema = ts.EMA(sgn)
    plt.title(label='Exponential Moving Average')
    plt.plot(sgn, label='sgn')
    plt.plot(ema, label='ema')
    plt.legend()
    plt.show()
    exit()
    #---------------------------------------------------------------------------

    # RSI indicator ------------------------------------------------------------
    lookback_size = 14
    hist = prices[-T*ndays-lookback_size:-T*ndays]
    sgn = np.concatenate((hist,days))
    rsi, lothreshold, hithreshold = ts.RSI(sgn,initsize=lookback_size)
    print(rsi)

    fig, axs = plt.subplots(2)
    axs[0].plot(days, label='prices')
    axs[0].legend()
    axs[1].plot(rsi, label=f'RSI lookback({lookback_size})')
    axs[1].plot(np.arange(0,len(rsi)), np.full(len(rsi),hithreshold), label=f'{hithreshold}', color='gray', linestyle='dashed')
    axs[1].plot(np.arange(0,len(rsi)), np.full(len(rsi),lothreshold), label=f'{lothreshold}', color='gray', linestyle='dashed')
    axs[1].legend()
    plt.show()
    #---------------------------------------------------------------------------
#===============================================================================
