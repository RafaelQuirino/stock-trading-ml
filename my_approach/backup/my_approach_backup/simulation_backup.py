import sys
import math
import datetime as dt
import time_series as ts

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/rafael/Documents/workspace/github/stock-trading-ml/my_approach')
import util

'''
'''
def strtodatetime (str):
    d = dt.datetime.strptime(str,"%Y-%m-%d %H:%M")
    return d.day, d.month, d.year, d.hour, d.minute

'''
TODO
'''
def datetimetostr (date):
    return 'TODO'

'''
'''
def save_point (x,xi, y,yi):
    x.append(xi)
    y.append(yi)

'''
'''
class Transaction:

    BUY  = 0
    SELL = 1

    def __init__(self,
        nstocks, ttype, tax,
        buy_datetime,  buy_price,
        sell_datetime, sell_price
    ):
        self.nstocks        = nstocks
        self.ttype           = ttype
        self.tax            = tax
        self.buy_datetime   = buy_datetime
        self.buy_price      = buy_price
        self.sell_datetime  = sell_datetime
        self.sell_price     = sell_price

    def print(self, treturn=0.0):
        if self.ttype == Transaction.BUY:
            print(f'--- Bought {self.nstocks} stocks by ${self.buy_price}, tax: ${self.tax}')
        elif self.ttype == Transaction.SELL:
            print(f'--- Sold {self.nstocks} stocks by ${self.sell_price}, tax: ${self.tax}, return: ${treturn}\n')

'''
'''
class Simulation:

    transaction_tax = 0.0308 * 1e-2    # : 0.0308%
    stop_loss       = 1.0000 * 1e-2
    stop_gain       = 1.5000 * 1e-2

    def __init__(self,
        # Data arguments
        prices, datetimes, file,
        # Trader arguments
        balance=0.0
    ):
        # Data
        self.file       = file
        self.prices     = prices
        self.datetimes  = datetimes

        # Simulation state
        self.current_timestep = 0
        self.current_datetime = None

        # Trader state
        self.balance         = balance
        self.nstocks         = 0
        self.stop_loss       = Simulation.stop_loss
        self.stop_gain       = Simulation.stop_gain
        self.transaction_tax = Simulation.transaction_tax
        self.transactions = []

    def buy (self, price):
        nstocks      = int(self.balance / price)
        trans_value  = nstocks*price
        trans_tax    = self.transactionTax(trans_value)
        if self.balance > trans_value:
            self.nstocks  = nstocks
            self.balance -= trans_value
            self.taxTransaction(trans_value)
            self.transactions.append(Transaction(nstocks, Transaction.BUY, trans_tax, None, price, None, None))
            self.transactions[-1].print()
        else:
            print('Not enough money')

    def sell (self, price):
        nstocks       = self.nstocks
        trans_value   = nstocks*price
        trans_tax     = self.transactionTax(trans_value)
        self.nstocks  = 0
        self.balance  += trans_value
        self.taxTransaction(trans_value)
        last = self.transactions[-1]
        treturn = (trans_value-trans_tax) - (last.nstocks * last.buy_price + last.tax)
        self.transactions.append(Transaction(nstocks, Transaction.SELL, trans_tax, None, None, None, price))
        self.transactions[-1].print(treturn=treturn)

    def transactionTax (self, transaction_value):
        return transaction_value * self.transaction_tax

    def taxTransaction (self, transaction_value):
        self.balance -= self.transactionTax(transaction_value)

    def simulate (self, start_datetime, end_datetime):
        return 0.0

    def plot_analysis (self, day_prices, buy_x, buy_y, sell_x, sell_y, sma1, sma2, loband, hiband, mm_norm):
        plt.plot(day_prices, label='prices')
        plt.plot(sma1, label='sma-recent')
        plt.plot(sma2, label='sma-older')
        plt.plot(loband, color='green', alpha=0.2)
        plt.plot(hiband, color='green', alpha=0.2)
        plt.plot(mm_norm, label='mm_norm', alpha=0.5, linestyle='dashed')
        plt.plot(np.arange(0,len(day_prices)), np.full(len(day_prices),(np.max(day_prices)-np.min(day_prices))/2.0)+np.min(day_prices), color='grey', alpha=0.2, linestyle='dashed')
        plt.scatter(buy_x, buy_y, marker='o', s=64.0, color='red', label='buy')
        plt.scatter(sell_x, sell_y, marker='o', s=64.0, color='green', label='sell')
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.legend()
        plt.show()


if __name__ == '__main__':

    import util
    import numpy as np
    import pandas as pd
    import datetime as dt
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    print('\nSimulation\n----------\n')

    # prices, datetimes, file = ts.import_data('data/vale_2019.csv', 'val1', 'datetime', 0)
    # prices, datetimes, file = ts.import_data('data/vale_2020.csv', 'preco', 'hora', 0)

    prices, datetimes, file = ts.import_data('data/vale3_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/abev3_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/bbas3_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/itub4_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/jbss3_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/petr4_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/petr4_2020.csv', 'preco', 'hora', 0)

    sim = Simulation(prices, datetimes, file, balance=1000.0)
    print(sim.file,'\n')
    print(sim.prices,'\n')
    print(sim.datetimes,'\n')

    #---------------------------------------------------------------------------
    # RUDIMENTAR SIMULATION
    #---------------------------------------------------------------------------
    IDLE           = 1
    IDLE_WAITING   = 2
    BOUGHT         = 4
    BOUGHT_WAITING = 8
    state  = IDLE

    _, loband, hiband = ts.bollinger_bands(sim.prices)

    print('BEGINNING SIMULATION')
    T     = 390*1
    ndays = int(float(len(sim.prices)) / float(T))

    balances = []

    # day = 255
    # for d in range(day,day+1):#ndays):
    for d in range(5,ndays):

        day_prices = sim.prices[d*T:(d+1)*T]
        old_prices = sim.prices[(d-5)*T:d*T]

        sma, loband, hiband = ts.bollinger_bands(day_prices, winsize=50, stdmult=2.2)
        sma20 = ts.moving_average(day_prices, winsize=20)
        sma50 = ts.moving_average(day_prices, winsize=50)
        winsize = int(len(day_prices)/16)
        mm = ts.get_momentum(day_prices, winsize=winsize, mtype=2, matype=1)
        mm_norm = ts.normalize_range(mm, np.min(day_prices), np.max(day_prices))

        mid = float(np.max(day_prices) - np.min(day_prices))/2.0 + np.min(day_prices)


        print('\n===============================================================')
        print(f'DAY: {d}')
        print('===============================================================\n')

        buy_x, buy_y = [], []
        sell_x, sell_y = [], []

        for i in range(len(day_prices)):

            price = day_prices[i]

            vec = np.concatenate((old_prices,day_prices[:i+1]))
            currmid = float(np.max(vec) - np.min(vec))/2.0 + np.min(vec)

            # Forcing sell in the end of the day -------------------------------
            if i == len(day_prices)-1:
                if state == BOUGHT or state == BOUGHT_WAITING:
                    sim.sell(price)
                    save_point(sell_x,i, sell_y,price)
                    state = IDLE
                    continue
            #-------------------------------------------------------------------

            # Implementing stop_loss strategy ----------------------------------
            if state == BOUGHT or state == BOUGHT_WAITING:
                last = sim.transactions[-1]
                spent = (last.nstocks*last.buy_price) + last.tax
                earning = (sim.nstocks*price)
                earning -= (earning * sim.transaction_tax)
                result = earning-spent
                if result < 0 and abs(result) > spent * sim.stop_loss:
                    sim.sell(price)
                    save_point(sell_x,i, sell_y,price)
                    state = IDLE
                    continue
            #-------------------------------------------------------------------

            # Implementing stop_gain strategy ----------------------------------
            if state == BOUGHT or state == BOUGHT_WAITING:
                last = sim.transactions[-1]
                spent = (last.nstocks*last.buy_price) + last.tax
                earning = (sim.nstocks*price)
                earning -= (earning * sim.transaction_tax)
                result = earning-spent
                # If reached stop_gain
                if result > 0 and result > spent * sim.stop_gain:
                    sim.sell(price)
                    save_point(sell_x,i, sell_y,price)
                    state = IDLE
                    continue
            #-------------------------------------------------------------------

            # Implementing momentum strategy -----------------------------------
            if state == IDLE_WAITING:
                if mm_norm[i]-currmid < -0.1:
                    if i < len(day_prices)-1:
                        sim.buy(price)
                        save_point(buy_x,i, buy_y,price)
                        state = BOUGHT
                        continue
            #-------------------------------------------------------------------

            # Implementing momentum strategy -----------------------------------
            if state == BOUGHT_WAITING:
                if mm_norm[i]-currmid > 0.1:
                    sim.sell(price)
                    save_point(sell_x,i, sell_y,price)
                    state = IDLE
                    continue
            #-------------------------------------------------------------------


            # Prepare do BUY ---------------------------------------------------
            if price < loband[i]:
                if state == IDLE:
                    state = IDLE_WAITING
            #-------------------------------------------------------------------

            # SELL -------------------------------------------------------------
            elif price > hiband[i]:
                if state == BOUGHT:
                    state = BOUGHT_WAITING
            #-------------------------------------------------------------------

        print(f'==> BALANCE: ${sim.balance}')
        balances.append(sim.balance)

        # sim.plot_analysis(day_prices, buy_x, buy_y, sell_x, sell_y, sma20, sma50, loband, hiband, mm_norm)

    plt.title(label=f'Simulating {sim.file}')
    plt.plot(balances, label='Amount')
    plt.legend()
    plt.show()
    #---------------------------------------------------------------------------
