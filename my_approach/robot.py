 #!/home/rafael/Documents/workspace/github/stock-trading-ml/venv/bin/python3

import math
import numpy as np
import simulation as s
import time_series as ts
import matplotlib.pyplot as plt

class Indicator:

    def __init__ (self, func=lambda: 0, args={}):
        self.x = 0

class Robot:

    # Parameters
    stop_loss       = 1.0
    stop_gain       = 1.5
    num_day_loss    = 3
    num_stop_loss   = 2
    analysis_window = 400 # Must be bigger than # of prices in a day

    # Possible actions
    ACTION_BUY  = 'BUY'
    ACTION_SELL = 'SELL'
    ACTION_WAIT = 'WAIT'

    # Possible states
    STATE_IDLE           = 'IDLE'
    STATE_IDLE_WAITING   = 'IDLE_WAITING'
    STATE_BOUGHT         = 'BOUGHT'
    STATE_BOUGHT_WAITING = 'BOUGHT_WAITING'

    '''
    '''
    def __init__(self, prices=[], datetimes=[], balance=0.0):
        # Robot information ----------------------------------------------------
        self.prices           = prices
        self.datetimes        = datetimes
        self.capital          = balance
        self.balance          = balance
        self.balances         = []
        self.returns          = []
        self.transactions     = []
        #_----------------------------------------------------------------------

        # Robot operation parameters -------------------------------------------
        self.state            = Robot.STATE_IDLE
        self.stop_loss        = Robot.stop_loss
        self.stop_gain        = Robot.stop_gain
        self.last_buy_price   = None
        self.current_price    = None
        self.current_datetime = None
        self.num_day_loss     = 0
        self.num_stop_loss    = 0
        self.day_lock         = False
        self.day_t0           = 0
        self.buy_x            = []
        self.buy_y            = []
        self.sell_x           = []
        self.sell_y           = []
        #-----------------------------------------------------------------------

        # Robot indicators -----------------------------------------------------
        self.initialized     = False
        self.bbwinsize       = 20
        self.bbstdmult       = 2.0
        self.minmax_scale    = 2 * ((1.0 + math.sqrt(5)) / 2.0)
        self.analysis_window = Robot.analysis_window // 2
        self.min             = None
        self.max             = None
        self.min_today       = None
        self.max_today       = None
        self.sma20           = None
        self.sma50           = None
        self.ema             = None
        self.bb              = None
        self.bb_peaks        = None
        self.mo20            = None
        self.mo50            = None
        self.rsi             = None
        self.cci             = None
        self.so              = None
        #-----------------------------------------------------------------------

    '''
    '''
    def init (self):
        self.min       = np.array([np.min(self.prices)]*len(self.prices))
        self.max       = np.array([np.max(self.prices)]*len(self.prices))
        self.min_today = np.array([np.min(self.prices)]*len(self.prices))
        self.max_today = np.array([np.max(self.prices)]*len(self.prices))
        self.smawin    = ts.SMA(self.prices, winsize=self.analysis_window)
        self.sma20     = ts.SMA(self.prices, winsize=20)
        self.sma50     = ts.SMA(self.prices, winsize=50)
        self.bb        = ts.BB(self.prices, winsize=self.bbwinsize, stdmult=self.bbstdmult)
        self.bb_peaks  = ts.BB_PEAKS_2(self.prices, self.bb)
        self.mo20      = ts.MO(self.prices, winsize=20)
        self.mo50      = ts.MO(self.prices, winsize=50)
        self.mo100     = ts.MO(self.prices, winsize=100)
        self.mo150     = ts.MO(self.prices, winsize=150)
        self.mo200     = ts.MO(self.prices, winsize=200)
        self.mo250     = ts.MO(self.prices, winsize=250)
        self.so20      = ts.SO(self.prices, winsize=20)
        self.so50      = ts.SO(self.prices, winsize=50)
        self.so100     = ts.SO(self.prices, winsize=100)
        self.initialized = True
        return self

    '''
    '''
    def updateIndicators (self):
        if self.initialized:
            AW = Robot.analysis_window
            p  = self.prices[-1]

            # Updating min and max
            scale = self.minmax_scale
            s = scale * AW if scale * AW < len(self.prices) else len(self.prices)
            s = int(s)
            self.min       = np.append(self.min, np.min(self.prices[-s:]))
            self.max       = np.append(self.max, np.max(self.prices[-s:]))
            self.min_today = np.append(self.min_today, np.min(self.prices[-self.day_t0:]))
            self.max_today = np.append(self.max_today, np.max(self.prices[-self.day_t0:]))

            # Update self.sma20
            analysis_window = self.analysis_window if self.analysis_window < len(self.prices) else len(self.prices)
            self.smawin = np.append(self.smawin, np.sum(self.prices[-analysis_window:]) / analysis_window)[-AW:]

            # Update self.sma20
            self.sma20 = np.append(self.sma20, np.sum(self.prices[-20:]) / 20.0)[-AW:]

            # Update self.sma50
            self.sma50 = np.append(self.sma50, np.sum(self.prices[-50:]) / 50.0)[-AW:]

            # Update self.bb20
            bbma, bblo, bbhi = self.bb[0], self.bb[1], self.bb[2]
            bbma = np.append(bbma, np.sum(self.prices[-self.bbwinsize:]) / self.bbwinsize)[-AW:]
            bblo = np.append(bblo, bbma[len(bbma)-1]-np.std(bbma[-self.bbwinsize:])*self.bbstdmult)[-AW:]
            bbhi = np.append(bbhi, bbma[len(bbma)-1]+np.std(bbma[-self.bbwinsize:])*self.bbstdmult)[-AW:]
            self.bb = (bbma,bblo,bbhi)

            # Update self.bb_peaks
            if p < self.bb[1][-1]:
                self.bb_peaks = np.append(self.bb_peaks, p - self.bb[1][-1])
            elif p > self.bb[2][-1]:
                self.bb_peaks = np.append(self.bb_peaks, p - self.bb[2][-1])
            else:
                self.bb_peaks = np.append(self.bb_peaks,0.0)

            # Update self.mo20
            mo, molim = self.mo20[0], self.mo20[1]
            mo = np.append(mo, (p/self.prices[-20])*molim)[-AW:]
            self.mo20 = (mo,molim)

            # Update self.mo50
            mo, molim = self.mo50[0], self.mo50[1]
            mo = np.append(mo, (p/self.prices[-50])*molim)[-AW:]
            self.mo50 = (mo,molim)

            # Update self.mo100
            mo, molim = self.mo100[0], self.mo100[1]
            mo = np.append(mo, (p/self.prices[-100])*molim)[-AW:]
            self.mo100 = (mo,molim)

            # Update self.mo150
            mo, molim = self.mo150[0], self.mo150[1]
            mo = np.append(mo, (p/self.prices[-150])*molim)[-AW:]
            self.mo150 = (mo,molim)

            # Update self.mo200
            mo, molim = self.mo200[0], self.mo200[1]
            mo = np.append(mo, (p/self.prices[-200])*molim)[-AW:]
            self.mo200 = (mo,molim)

            # Update self.mo250
            mo, molim = self.mo250[0], self.mo250[1]
            mo = np.append(mo, (p/self.prices[-250])*molim)[-AW:]
            self.mo250 = (mo,molim)

            # Update self.so20
            so, solo, sohi = self.so20[0], self.so20[1], self.so20[2]
            lo = np.min(self.prices[-20:])
            hi = np.max(self.prices[-20:])
            so = np.append(so, ((p-lo)/(hi-lo))*100.0)[-AW:]
            self.so20 = (so,solo,sohi)

            # Update self.so50
            so, solo, sohi = self.so50[0], self.so50[1], self.so50[2]
            lo = np.min(self.prices[-50:])
            hi = np.max(self.prices[-50:])
            so = np.append(so, ((p-lo)/(hi-lo))*100.0)[-AW:]
            self.so50 = (so,solo,sohi)

            # Update self.so100
            so, solo, sohi = self.so100[0], self.so100[1], self.so100[2]
            lo = np.min(self.prices[-100:])
            hi = np.max(self.prices[-100:])
            so = np.append(so, ((p-lo)/(hi-lo))*100.0)[-AW:]
            self.so100 = (so,solo,sohi)

            return True
        else:
            return False

    '''
    '''
    def setHistory (self, history):
        self.prices    = np.array(history[0])
        self.datetimes = np.array(history[1])
        return self

    '''
    '''
    def showNewData (self, newprice, newdatetime):
        self.prices = np.append(self.prices, newprice)
        self.current_price = newprice
        self.datetimes = np.append(self.datetimes, newdatetime)
        if self.current_datetime != None:
            if newdatetime.day != self.current_datetime.day:
                self.day_t0 = len(self.datetimes)-1
                self.num_day_loss  = 0
                self.num_stop_loss = 0
                self.day_lock      = False
                self.buy_x         = []
                self.buy_y         = []
                self.sell_x        = []
                self.sell_y        = []
        self.current_datetime = newdatetime
        self.updateIndicators()

    '''
    '''
    def forecast (self, winsize=1):
        prediction = [0.0] * winsize
        return prediction

    '''
    '''
    def buy (self, price, datetime, simulation, printTransaction=False):
        nstocks = max(int(simulation.balance/price)-1, 0)
        success = simulation.buy(price, datetime, nstocks, printTransaction)
        self.balance = simulation.balance
        if success:
            self.transactions.append(simulation.transactions[len(simulation.transactions)-1])
            self.state = Robot.STATE_BOUGHT
            self.last_buy_price = price
            return True
        else:
            print(f'Could not buy {nstocks} stocks at ${price} each.')
            return False

    '''
    '''
    def sell (self, price, datetime, simulation, printTransaction=False):
        nstocks = simulation.nstocks
        treturn = simulation.sell(price, datetime, nstocks, printTransaction)
        self.balance = simulation.balance
        if treturn != None:
            self.transactions.append(simulation.transactions[len(simulation.transactions)-1])
            self.balances.append(self.balance)
            self.returns.append(treturn)
            self.state = Robot.STATE_IDLE
            self.last_buy_price = None

            if treturn < 0:
                self.num_day_loss += 1
                if self.num_day_loss >= Robot.num_day_loss:
                    self.day_lock = True
                    print('>>> DAY LOCKED')

        return treturn

    '''
    '''
    def query (self, newprice=None, newdatetime=None):
        # Updating robot information with new observations ---------------------
        if newprice != None :
            self.showNewData(newprice, newdatetime)
        p = self.prices[len(self.prices)-1]
        d = self.datetimes[len(self.datetimes)-1]
        #-----------------------------------------------------------------------

        # Inline indicators ----------------------------------------------------
        # Period for the last day ------------------------------
        T = 0
        baseday = self.datetimes[len(self.datetimes)-2].day
        while True:
            T += 1
            day = self.datetimes[len(self.datetimes)-2-T].day
            if day != baseday:
                break
        #-------------------------------------------------------

        # Ensuring correcteness -----
        T   = min(T,len(self.min))
        sgn = self.prices[-T:-1]
        l   = len(sgn)
        #----------------------------

        # Today's mins and maxes ------------------------------------
        todaymin, todaymax = [], []
        for i in range(l):
            todaymin.append(min(sgn[0:i+1]))
            todaymax.append(max(sgn[0:i+1]))
        todaymin, todaymax = np.array(todaymin), np.array(todaymax)
        self.min_today = todaymin
        self.max_today = todaymax
        #------------------------------------------------------------
        #-----------------------------------------------------------------------

        # Plotting daily analysis ----------------------------------------------
        if True:
            if self.datetimes[-1].day != self.datetimes[-2].day:
                # # Period for the last day
                # T = 0
                # baseday = self.datetimes[len(self.datetimes)-2].day
                # while True:
                #     T += 1
                #     day = self.datetimes[len(self.datetimes)-2-T].day
                #     if day != baseday:
                #         break

                # # Ensuring correcteness
                # T   = min(T,len(self.min))
                # sgn = self.prices[-T:-1]
                # l   = len(sgn)
                # todaymin, todaymax = [], []
                # for i in range(l):
                #     todaymin.append(min(sgn[0:i+1]))
                #     todaymax.append(max(sgn[0:i+1]))
                # todaymin, todaymax = np.array(todaymin), np.array(todaymax)

                fig, axs = plt.subplots(4)
                axs[0].plot(sgn, linewidth=3)
                axs[0].scatter(self.buy_x[-T:-1], self.buy_y[-T:-1], color='red')
                print(self.buy_x)
                print(self.buy_y)
                # exit()
                # axs[0].plot(np.arange(0,len(self.min[-T:-1])), self.min[-T:-1], label=f'minwin: {self.min[-1]}', linestyle='dashed')
                # axs[0].plot(np.arange(0,len(self.max[-T:-1])), self.max[-T:-1], label=f'maxwin: {self.max[-1]}', linestyle='dashed')
                axs[0].plot(np.arange(0,len(self.min_today[-T:-1])), self.min_today[-T:-1], label=f'min today', linestyle='dashed', alpha=0.4)
                axs[0].plot(np.arange(0,len(self.max_today[-T:-1])), self.max_today[-T:-1], label=f'max today', linestyle='dashed', alpha=0.4)
                axs[0].plot(self.sma20[-T:-1], label=f'SMA20')
                axs[0].plot(self.sma50[-T:-1], label='SMA50')
                axs[0].plot(self.smawin[-T:-1], label='SMAWIN')
                axs[0].plot(self.bb[0][-T:-1], color='red', alpha=0.5, linewidth=1)
                axs[0].plot(self.bb[1][-T:-1], color='green', alpha=0.7, linewidth=1)
                axs[0].plot(self.bb[2][-T:-1], color='green', alpha=0.7, linewidth=1)
                axs[0].fill_between(np.arange(0,len(sgn)), self.bb[1][-T:-1], self.bb[2][-T:-1], color='green', alpha=0.1)
                axs[0].legend()

                # bb_peaks
                axs[1].plot(np.arange(0,len(self.bb_peaks[-T:-1])), np.full(len(self.bb_peaks[-T:-1]),0), color='gray', linestyle='dashed')
                axs[1].plot(self.bb_peaks[-T:-1], label='BB_PEAKS')
                axs[1].plot(ts.weighted_moving_average(self.bb_peaks[-T:-1],winsize=20), label='BB_PEAKS')
                axs[1].plot(ts.weighted_moving_average(self.bb_peaks[-T:-1],winsize=50), label='BB_PEAKS')
                axs[1].plot(ts.weighted_moving_average(self.bb_peaks[-T:-1],winsize=20)+ts.weighted_moving_average(self.bb_peaks[-T:-1],winsize=50), label='BB_PEAKS')
                axs[1].legend()

                axs[2].plot(np.arange(0,len(self.mo20[0][-T:-1])), np.full(len(self.mo20[0][-T:-1]),self.mo20[1]), label=f'{self.mo20[1]}', color='gray', linestyle='dashed')
                axs[2].plot(ts.SMA(self.mo20[0][-T:-1]), label='MO20')
                axs[2].plot(ts.SMA(self.mo50[0][-T:-1]), label='MO50')
                axs[2].plot(ts.SMA(self.mo100[0][-T:-1]), label='MO100')
                axs[2].plot(ts.SMA(self.mo150[0][-T:-1]), label='MO150')
                axs[2].plot(ts.SMA(self.mo200[0][-T:-1]), label='MO200')
                # axs[1].plot(ts.SMA(self.mo250[0][-T:-1]), label='MO250')
                axs[2].legend()

                # axs[2].plot(np.arange(0,len(self.so20[0][-T:-1])), np.full(len(self.so20[0][-T:-1]),self.so20[2]), label=f'{self.so20[2]}', color='gray', linestyle='dashed')
                # axs[2].plot(np.arange(0,len(self.so20[0][-T:-1])), np.full(len(self.so20[0][-T:-1]),self.so20[1]), label=f'{self.so20[1]}', color='gray', linestyle='dashed')
                # # axs[2].plot(self.so20[0][-T:-1], label='SO20')
                # axs[2].plot(ts.SMA(self.so20[0][-T:-1], winsize=20), label='SMA20(SO20)')
                # axs[2].plot(ts.SMA(self.so20[0][-T:-1], winsize=50), label='SMA50(SO20)')
                # axs[2].plot(ts.SMA(self.so20[0][-T:-1], winsize=100), label='SMA100(SO20)')
                # axs[2].legend()

                axs[3].plot(np.arange(0,len(self.so20[0][-T:-1])), np.full(len(self.so20[0][-T:-1]),self.so20[2]), label=f'{self.so20[2]}', color='gray', linestyle='dashed')
                axs[3].plot(np.arange(0,len(self.so20[0][-T:-1])), np.full(len(self.so20[0][-T:-1]),self.so20[1]), label=f'{self.so20[1]}', color='gray', linestyle='dashed')
                # axs[3].plot(self.so20[0][-T:-1], label='SO20')
                axs[3].plot(ts.SMA(self.so20[0][-T:-1]), label='SO20')
                axs[3].plot(ts.SMA(self.so50[0][-T:-1]), label='SO50')
                axs[3].plot(ts.SMA(self.so100[0][-T:-1]), label='SO100')
                axs[3].legend()

                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                plt.show()
        #-----------------------------------------------------------------------

        if not self.day_lock:
            # Implementing stop loss strategy ----------------------------------
            if self.last_buy_price != None:
                diff = p - self.last_buy_price
                if diff < 0:
                    percent = (abs(diff) * 100.0) / self.last_buy_price
                    if percent >= self.stop_loss:
                        print('>>> STOP LOSS SELL')
                        self.num_stop_loss += 1
                        if self.num_stop_loss >= Robot.num_stop_loss:
                            self.day_lock = True
                            print('>>> DAY LOCKED')
                        return Robot.ACTION_SELL
            #-------------------------------------------------------------------

            # Implementing stop gain strategy ----------------------------------
            if self.last_buy_price != None:
                diff = p - self.last_buy_price
                if diff > 0:
                    percent = (diff * 100.0) / self.last_buy_price
                    if percent >= self.stop_gain:
                        print('>>> STOP GAIN SELL')
                        return Robot.ACTION_SELL
            #-------------------------------------------------------------------

            #-------------------------------------------------------------------
            # ROBOT STRATEGY
            #-------------------------------------------------------------------
            # if abs(p - self.bb[1][-1]) > 0.05:
            #     if self.mo20[0][-1] < self.mo20[1] and self.mo50[0][-1] < self.mo50[1] and self.mo100[0][-1] < self.mo100[1]:
            #         if self.so20[0][-1] < self.so20[1] and self.so50[0][-1] < self.so50[1] and self.so100[0][-1] < self.so100[1]:
            #             if self.state == Robot.STATE_IDLE or self.state == Robot.STATE_IDLE_WAITING:
            #                 return Robot.ACTION_BUY

            if len(todaymin) > 30:
                if abs(p-self.bb[1][-1]) > 0.05 and abs(p-todaymin[-1]) > 0.02:
                    if self.mo20[0][-1] < self.mo20[1] and self.mo50[0][-1] < self.mo50[1] and self.mo100[0][-1] < self.mo100[1]:
                        if self.state == Robot.STATE_IDLE or self.state == Robot.STATE_IDLE_WAITING:
                            self.buy_x.append(len(self.prices)-1 - self.day_t0 + 1)
                            self.buy_y.append(p)
                            print(len(self.prices)-1 - self.day_t0 + 1, p)
                            return Robot.ACTION_BUY

                if self.last_buy_price != None:
                    if p >= self.last_buy_price + 0.05 or p > self.bb[2][-1] + 0.2:
                        if self.state == Robot.STATE_BOUGHT or self.state == Robot.STATE_BOUGHT_WAITING:
                            return Robot.ACTION_SELL
            #-------------------------------------------------------------------

        return Robot.ACTION_WAIT

#===============================================================================
# +-------+
# | TESTS |
# +-------+
#===============================================================================
if __name__ == '__main__':
    x = 0
#===============================================================================
