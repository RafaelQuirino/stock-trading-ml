import sys
import math
import datetime as dt
import time_series as ts
import matplotlib.pyplot as plt

'''
+-------------------+
| Transaction class |
+-------------------+
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
+------------------+
| Simulation class |
+------------------+
'''
class Simulation:

    '''
    Transaction tax, currently 0.0308% of each transaction (buy or sell)
    '''
    transaction_tax = 0.0308 * 1e-2

    '''
    Simulation constructor
    '''
    def __init__(self,
        # Data arguments
        prices, datetimes, file,
        # Trading arguments
        balance=0.0, nstocks=0
    ):
        # Data
        self.file       = file
        self.prices     = prices
        self.datetimes  = datetimes

        # Simulation state
        self.current_timestep = 0
        self.current_datetime = None

        # Trading state
        self.balance         = balance
        self.nstocks         = nstocks
        self.transaction_tax = Simulation.transaction_tax
        self.transactions = []

    '''
    Just calculates the tax
    '''
    def transactionTax (self, transaction_value):
        return transaction_value * self.transaction_tax

    '''
    Calculate, and apply the tax
    '''
    def taxTransaction (self, transaction_value):
        self.balance -= self.transactionTax(transaction_value)

    '''
    Buy as much stocks as self.balance can
    '''
    def buy (self, price, printTransaction=False):
        nstocks      = int(self.balance / price)
        trans_value  = nstocks*price
        trans_tax    = self.transactionTax(trans_value)
        if self.balance > trans_value:
            self.nstocks  = nstocks
            self.balance -= trans_value
            self.taxTransaction(trans_value)
            self.transactions.append(Transaction(nstocks, Transaction.BUY, trans_tax, None, price, None, None))
            if printTransaction:
                self.transactions[-1].print()
        else:
            print('Not enough money.')

    '''
    Sell all stocks currently being held
    '''
    def sell (self, price, printTransaction=False):
        nstocks       = self.nstocks
        trans_value   = nstocks*price
        trans_tax     = self.transactionTax(trans_value)
        self.nstocks  = 0
        self.balance  += trans_value
        self.taxTransaction(trans_value)
        last = self.transactions[-1]
        treturn = (trans_value-trans_tax) - (last.nstocks * last.buy_price + last.tax)
        self.transactions.append(Transaction(nstocks, Transaction.SELL, trans_tax, None, None, None, price))
        if printTransaction:
            self.transactions[-1].print(treturn=treturn)

    '''
    Simulate operation, day by day, in the given datetime range
    '''
    def simulate (self, start_datetime, end_datetime):
        return 0.0

    '''
    '''
    def plot_analysis (self, day_prices, buy_x, buy_y, sell_x, sell_y, sma, ema):#, loband, hiband):
        plt.plot(day_prices, label='prices')
        plt.plot(sma, label='sma')
        plt.plot(ema, label='ema')
        # plt.plot(loband, color='green', alpha=0.2)
        # plt.plot(hiband, color='green', alpha=0.2)
        plt.scatter(buy_x, buy_y, marker='o', s=64.0, color='red', label='buy')
        plt.scatter(sell_x, sell_y, marker='o', s=64.0, color='green', label='sell')
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.legend()
        plt.show()



'''
'''
def save_point (x,xi, y,yi):
    x.append(xi)
    y.append(yi)

if __name__ == '__main__':

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
    state          = IDLE

    T        = 390*1
    ndays    = int(float(len(sim.prices)) / float(T))
    balances = []

    day = 252
    for d in range(day,day+1):#ndays):
    # for d in range(ndays):

        print('\n===============================================================')
        print(f'DAY: {d}')
        print('===============================================================\n')

        day_prices = sim.prices[d*T:(d+1)*T]
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []

        for i in range(len(day_prices)):

            price = day_prices[i]

            sma = ts.SMA(day_prices[:i+1])
            ema = ts.EMA(day_prices[:i+1])

            # Forcing sell in the end of the day -------------------------------
            if i == len(day_prices)-1:
                if state == BOUGHT or state == BOUGHT_WAITING:
                    sim.sell(price, printTransaction=True)
                    save_point(sell_x,i, sell_y,price)
                    state = IDLE
                    continue
            #-------------------------------------------------------------------

            # if sma[i] < ema[i]:
            if ema[i] - sma[i] > 0.1:
                if state == IDLE:
                    sim.buy(price, printTransaction=True)
                    save_point(buy_x,i, buy_y,price)
                    state = BOUGHT
                    continue

            # elif ema[i] < sma[i]:
            elif sma[i] - ema[i] > 0.1:
                if state == BOUGHT:
                    sim.sell(price, printTransaction=True)
                    save_point(sell_x,i, sell_y,price)
                    state = IDLE
                    continue

        print(f'==> BALANCE: ${sim.balance}')
        balances.append(sim.balance)
        sim.plot_analysis (day_prices, buy_x, buy_y, sell_x, sell_y, sma, ema)

    # Showing balance after simulation finished
    plt.title(label=f'Simulating {sim.file}')
    plt.plot(balances, label='Amount')
    plt.legend()
    plt.show()
