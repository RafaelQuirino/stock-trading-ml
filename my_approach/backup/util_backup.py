import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#===============================================================================
'''
THIS CLASS ACTUALLY WORKS !
A class to make matplotlib graphs zoomable and panable with the mouse
'''
#===============================================================================
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
#===============================================================================

'''
'''
def load_col_from_csv(csvpath, colname):
    df = pd.read_csv(csvpath)
    return np.array(df[colname])

'''
'''
def minmax_normalization(sgn):
    sgn_normalizer = preprocessing.MinMaxScaler()
    sgn_normalized = sgn_normalizer.fit_transform(sgn)
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
    # for i in range(len(sgn)):
    #     a, b = i-winsize+1, i+1
    #     if i < winsize - 1:
    #         a, b = 0, i+1
    #
    #     # w = np.arange(1,(b-a)+1)
    #     # v = w * sgn[a:b]
    #     # ma[i] = v.sum() / w.sum()

    # n = len(sgn)
    # ema = np.zeros(n)
    # ema[0] = sgn[0]
    # for i in range(1,n):
    #     ema[i] = sgn[i]*(2.0/(1.0+n)) + ema[i-1]*(1-(2.0/(1.0+n)))

    ema = np.zeros(len(sgn))
    for i in range(len(sgn)):
        a, b = i-winsize+1, i+1
        if i < winsize - 1:
            a, b = 0, i+1

        n = b-a

        # alpha = 1.0/(n+1)
        # w = np.arange(0,n)[::-1]
        # w = np.apply_along_axis(lambda x: (1-alpha)**x, 0, w)
        # plt.plot(w,label='smoothing=1.0')
        #
        # alpha = 2.0/(n+1)
        # w = np.arange(0,n)[::-1]
        # w = np.apply_along_axis(lambda x: (1-alpha)**x, 0, w)
        # plt.plot(w,label='smoothing=2.0')
        #
        # alpha = 3.0/(n+1)
        # w = np.arange(0,n)[::-1]
        # w = np.apply_along_axis(lambda x: (1-alpha)**x, 0, w)
        # plt.plot(w,label='smoothing=3.0')

        alpha = smoothing/(n+1)
        w = np.arange(0,n)[::-1]
        w = np.apply_along_axis(lambda x: (1-alpha)**x, 0, w)
        # plt.plot(w,label='smoothing=4.0')

        # plt.legend()
        # plt.show()
        # exit()

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
def plot_bollinger_bands(sgn, ax, winsize=20, stdmult=2.0, autoadjust=False):
    if autoadjust:
        winsize = int(len(sgn) / 10.0)
    ma, loband, hiband = bollinger_bands(sgn, winsize, stdmult)

    # fig, ax = plt.subplots()
    ax.plot(sgn)
    ax.plot(ma, color='red', alpha=0.5, linewidth=1)
    ax.plot(loband, color='green', alpha=0.7, linewidth=1)
    ax.plot(hiband, color='green', alpha=0.7, linewidth=1)
    ax.fill_between(np.arange(0,len(sgn)),loband, hiband, color='green', alpha=0.1)
    # zp = ZoomPan(ax, base_scale=1.1)
    # plt.show()
    # zp.destroy()
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

'''
'''
def plot_analysis(
    sgn, sma_short, sma_long, ma, loband, hiband,
    balance, buy_x, buy_y, sell_x, sell_y
):
    fig, ax = plt.subplots()
    plt.title(f'Balance: {balance}')
    ax.plot(sgn, label='SIGNAL')
    ax.plot(sma_short, label='SMA-SHORT')
    ax.plot(sma_long, label='SMA-LONG')
    ma, loband, hiband = bollinger_bands(sgn, winsize=20, stdmult=2.0)
    ax.plot(ma, color='green', linewidth=1, alpha=0.2, linestyle='dashed')
    ax.plot(loband, color='green', linewidth=1, alpha=0.2)
    ax.plot(hiband, color='green', linewidth=1, alpha=0.2)
    ax.fill_between(np.arange(0,len(sgn)),loband, hiband, color='green', alpha=0.1)
    ax.scatter(buy_x, buy_y, marker='o', s=64.0, color='red', label='buy')
    ax.scatter(sell_x, sell_y, marker='o', s=64.0, color='green', label='sell')
    zp = ZoomPan(ax, base_scale=1.1)
    plt.legend()
    plt.show()
    zp.destroy()

def trading_strategy_1(sgn):
    # Tracking the state
    IDLE = 0
    BOUGHT = 1
    state = IDLE

    balance = 0.0
    last_bought = sgn[0]

    sma20 = moving_average(sgn, winsize=20)
    sma50 = moving_average(sgn, winsize=50)
    ma, loband, hiband = bollinger_bands(sgn, winsize=20, stdmult=2.0)

    buy_x = []
    buy_y = []
    sell_x = []
    sell_y = []
    for i in range(len(sgn)):
        if sma50[i] > sma20[i]:
            if state == IDLE:
                buy_x.append(i)
                buy_y.append(sgn[i])
                balance -= sgn[i]
                last_bought = sgn[i]
                state = BOUGHT
        elif sma20[i] > sma50[i]:
            if state == BOUGHT:
                sell_x.append(i)
                sell_y.append(sgn[i])
                balance += sgn[i]
                state = IDLE

    plot_analysis(
        sgn, sma20, sma50, ma, loband, hiband,
        balance, buy_x, buy_y, sell_x, sell_y
    )

    print('balance with sma_short / sma_long strategy: ', balance)

def trading_strategy_1_1(sgn):
    # Tracking the state
    IDLE = 0
    BOUGHT = 1
    state = IDLE

    balance = 0.0
    last_bought = sgn[0]

    sma20 = moving_average(sgn, winsize=20)
    sma50 = moving_average(sgn, winsize=50)
    ma, loband, hiband = bollinger_bands(sgn, winsize=20, stdmult=2.0)

    buy_x = []
    buy_y = []
    sell_x = []
    sell_y = []
    for i in range(len(sgn)):
        if sma50[i] > sma20[i]:
            if state == IDLE:
                buy_x.append(i)
                buy_y.append(sgn[i])
                balance -= sgn[i]
                last_bought = sgn[i]
                state = BOUGHT
        elif sma20[i] > sma50[i]:
            if state == BOUGHT and sgn[i] > last_bought:
                sell_x.append(i)
                sell_y.append(sgn[i])
                balance += sgn[i]
                state = IDLE

    plot_analysis(
        sgn, sma20, sma50, ma, loband, hiband,
        balance, buy_x, buy_y, sell_x, sell_y
    )

    print('balance with sma_short / sma_long strategy (and avoid losing): ', balance)

'''
BOLLINGER BANDS STRATEGY
'''
def trading_strategy_2(sgn):
    # Tracking the state
    IDLE = 0
    BOUGHT = 1
    state = IDLE

    balance = 0.0
    last_bought = sgn[0]

    # sma20 = moving_average(sgn, winsize=20)
    # sma50 = moving_average(sgn, winsize=50)
    ma, loband, hiband = bollinger_bands(sgn, winsize=20, stdmult=2.0)

    buy_x = []
    buy_y = []
    sell_x = []
    sell_y = []
    for i in range(len(sgn)):
        if sgn[i] < loband[i]:
            if state == IDLE:
                buy_x.append(i)
                buy_y.append(sgn[i])
                balance -= sgn[i]
                last_bought = sgn[i]
                state = BOUGHT
        elif sgn[i] > hiband[i]:
            if state == BOUGHT: # and sgn[i] > last_bought:
                sell_x.append(i)
                sell_y.append(sgn[i])
                balance += sgn[i]
                state = IDLE

    plot_analysis(
        sgn, [], [], ma, loband, hiband,
        balance, buy_x, buy_y, sell_x, sell_y
    )

    print('balance with bollinger_bands strategy: ', balance)

'''
BOLLINGER BANDS STRATEGY, WIDTH ANALYSIS
'''
def trading_strategy_3(sgn, apport=0.0, use_stop_loss=True, use_stop_gain=True):
    # Tracking the state
    IDLE = 0
    BOUGHT = 1
    state = IDLE

    amount           = apport
    balance          = 0.0
    nstocks          = 0
    last_bought      = 1e32
    stop_loss        = 1.0      # 1.0%
    stop_gain        = 1.5      # 1.5%
    transaction_rate = 0.0308 * 1e-2 # 0.0308%

    # sma20 = moving_average(sgn, winsize=20)
    # sma50 = moving_average(sgn, winsize=50)
    import math
    ma, loband, hiband = bollinger_bands(sgn, winsize=64, stdmult=2.0*1.618033988749895)
    bbwidth = hiband-loband
    bbgrads = np.gradient(bbwidth) # derivatives of elements (gradients)

    # print(bbwidth)
    # print(bbgrads)
    # plt.plot(bbwidth)
    # plt.plot(bbgrads)
    # plt.show()
    # exit()

    buy_x = []
    buy_y = []
    sell_x = []
    sell_y = []
    for i in range(1,len(sgn)):
        #-----------------------------------------------------------------------
        # Force the selling if in end of the day or
        # if losses dropped below stop_loss or
        # gains go above stop_gain
        #-----------------------------------------------------------------------
        if use_stop_loss and not use_stop_gain:
            if (i == len(sgn)-1 and state == BOUGHT) or \
                (((last_bought-sgn[i]) > 0 and ((last_bought-sgn[i])*100.0)/last_bought >= stop_loss) and state == BOUGHT):
                # Selling -------------
                transaction_val = nstocks * sgn[i]
                amount += transaction_val
                amount -= (transaction_val*transaction_rate)
                nstocks = 0

                sell_x.append(i)
                sell_y.append(sgn[i])
                balance += sgn[i]
                state = IDLE
                #----------------------
                continue

        if use_stop_gain and not use_stop_loss:
            if (i == len(sgn)-1 and state == BOUGHT) or \
                (((sgn[i]-last_bought) > 0 and ((sgn[i]-last_bought)*100.0)/last_bought >= stop_gain) and state == BOUGHT):
                # Selling -------------
                transaction_val = nstocks * sgn[i]
                amount += transaction_val
                amount -= (transaction_val*transaction_rate)
                nstocks = 0

                sell_x.append(i)
                sell_y.append(sgn[i])
                balance += sgn[i]
                state = IDLE
                #----------------------
                continue

        if use_stop_loss and use_stop_gain:
            if (i == len(sgn)-1 and state == BOUGHT) or \
                (((last_bought-sgn[i]) > 0 and ((last_bought-sgn[i])*100.0)/last_bought >= stop_loss) and state == BOUGHT) or \
                (((sgn[i]-last_bought) > 0 and ((sgn[i]-last_bought)*100.0)/last_bought >= stop_gain) and state == BOUGHT):
                # Selling -------------
                transaction_val = nstocks * sgn[i]
                amount += transaction_val
                amount -= (transaction_val*transaction_rate)
                nstocks = 0

                sell_x.append(i)
                sell_y.append(sgn[i])
                balance += sgn[i]
                state = IDLE
                #----------------------
                continue

        else:
            if (i == len(sgn)-1 and state == BOUGHT):
                # Selling -------------
                transaction_val = nstocks * sgn[i]
                amount += transaction_val
                amount -= (transaction_val*transaction_rate)
                nstocks = 0

                sell_x.append(i)
                sell_y.append(sgn[i])
                balance += sgn[i]
                state = IDLE
                #----------------------
                continue
        #-----------------------------------------------------------------------

        # +-----+
        # | BUY |
        # +-----+
        if sgn[i] < loband[i] and i < len(sgn)-10:
            if state == IDLE:
                if loband[i] >= loband[i-1] and bbgrads[i] < 0.5:
                    # Buying --------------
                    if amount > sgn[i]:
                        nstocks = amount / sgn[i]
                        transaction_val = nstocks * sgn[i]
                        amount -= transaction_val
                        amount -= (transaction_val*transaction_rate)

                        buy_x.append(i)
                        buy_y.append(sgn[i])
                        balance -= sgn[i]
                        last_bought = sgn[i]
                        state = BOUGHT
                    #----------------------

        # +------+
        # | SELL |
        # +------+
        if sgn[i] > hiband[i]:
            if state == BOUGHT:
                # Selling -------------
                transaction_val = nstocks * sgn[i]
                amount += transaction_val - (transaction_val*transaction_rate)
                nstocks = 0

                sell_x.append(i)
                sell_y.append(sgn[i])
                balance += sgn[i]
                state = IDLE
                #----------------------

    if False and balance < -2.0:
        plot_analysis(
            sgn, [], [], ma, loband, hiband,
            balance, buy_x, buy_y, sell_x, sell_y
        )

    print('balance with bollinger_bands strategy: ', balance, '\t\t capital: $', amount)
    return balance, amount

'''
=> Main function
'''
if __name__ == '__main__':
    file = 'data/vale_2019.csv' #'data/vale_2019.csv'
    column = 'val1' #'val1'
    sgn = load_col_from_csv(file,column)
    # plot(sgn)
    # exit()

    T = 390
    # sgn = sgn[-2*T:-T]

    # winsize = 50
    # plt.plot(sgn, label='SGN')
    # plt.plot(moving_average(sgn,winsize), label='SMA')
    # plt.plot(weighted_moving_average(sgn,winsize), label='WMA')
    # plt.plot(exponential_moving_average(sgn,winsize), label='EMA')
    # plt.legend()
    # plt.show()

    # plot_bollinger_bands(sgn, winsize=20, stdmult=2.0, autoadjust=False)

    print(len(sgn))
    print('mean: ', sample_mean(sgn))
    print('variance: ', sample_variance(sgn))



    # trading_strategy_1(sgn)
    # trading_strategy_2(sgn)

    apport = 1000.0
    balance      = 0
    balances     = [0]
    cum_balances = [0]
    capital      = [apport]
    ndays    = int(float(len(sgn)) / float(T))
    for i in range(ndays):
        b, m = trading_strategy_3(sgn[i*T:(i+1)*T], apport=apport, use_stop_loss=True, use_stop_gain=True)
        apport = m
        balance += b
        balances.append(b)
        cum_balances.append(balance)
        capital.append(apport)
    print('Total balance: ', balance)

    plt.plot(balances, label='returns')
    plt.plot(cum_balances, label='cum. returns')
    plt.legend()
    plt.show()
    plt.plot(capital, label='capital')
    plt.legend()
    plt.show()
