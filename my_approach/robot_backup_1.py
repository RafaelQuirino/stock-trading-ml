#!/home/rafael/Documents/workspace/github/stock-trading-ml/venv/bin/python3

import numpy as np
import simulation as s
import time_series as ts
import matplotlib.pyplot as plt

class Indicator:

    def __init__ (self, func=lambda: 0, args={}):
        self.x = 0

class Robot:

    # Parameters
    stop_loss    = 0.5
    stop_gain    = 2.0
    num_day_loss = 3

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
        self.prices           = prices
        self.datetimes        = datetimes
        self.balance          = balance
        self.balances         = []
        self.returns          = []
        self.transactions     = []
        self.state            = Robot.STATE_IDLE
        self.stop_loss        = Robot.stop_loss
        self.stop_gain        = Robot.stop_gain
        self.last_buy_price   = None
        self.current_price    = None
        self.current_datetime = None
        self.num_day_loss     = 0
        self.day_lock         = False

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
                self.num_day_loss = 0
                self.day_lock     = False
        self.current_datetime = newdatetime

    '''
    '''
    def updateIndicators (self):
        return 0

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
    def init (self):
        return 0

    '''
    '''
    def query (self, newprice=None, newdatetime=None):
        if newprice != None :
            self.showNewData(newprice, newdatetime)

        # Current price
        p = self.prices[len(self.prices)-1]

        if not self.day_lock:
            # Implementing stop loss strategy --------------------------------------
            if self.last_buy_price != None:
                diff = p - self.last_buy_price
                if diff < 0:
                    percent = (abs(diff) * 100.0) / self.last_buy_price
                    if percent >= self.stop_loss:
                        print('>>> STOP LOSS SELL')
                        return Robot.ACTION_SELL
            #-----------------------------------------------------------------------

            # Implementing stop gain strategy --------------------------------------
            if self.last_buy_price != None:
                diff = p - self.last_buy_price
                if diff > 0:
                    percent = (diff * 100.0) / self.last_buy_price
                    if percent >= self.stop_gain:
                        print('>>> STOP GAIN SELL')
                        return Robot.ACTION_SELL
            #-----------------------------------------------------------------------

            T = 420
            sma50 = ts.SMA(self.prices[-T:], winsize=50)
            sma20, bblo, bbhi = ts.BB(self.prices[-T:])
            m20, m50 = sma20[len(sma20)-1], sma50[len(sma50)-1]
            bbl, bbh = bblo[len(bblo)-1], bbhi[len(bbhi)-1]

            if self.state == Robot.STATE_IDLE:
                if p < bbl:
                    self.state = Robot.STATE_IDLE_WAITING

            elif self.state == Robot.STATE_IDLE_WAITING:
                if m20 < m50:
                    return Robot.ACTION_BUY

            elif self.state == Robot.STATE_BOUGHT:
                if p > bbh:
                    return Robot.ACTION_SELL

        return Robot.ACTION_WAIT

#===============================================================================
# +-------+
# | TESTS |
# +-------+
#===============================================================================
if __name__ == '__main__':
    x = 0
#===============================================================================
