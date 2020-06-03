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
    TRANSACTION_TAX = 0.0308 * 1e-2

    '''
    Simulation state
    '''
    IDLE   = 0
    BOUGHT = 1

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
        self.transaction_tax = Simulation.TRANSACTION_TAX
        self.transactions    = []
        self.net_return      = 0.0
        self.state           = Simulation.IDLE

    '''
    Just calculates the tax
    '''
    def transactionTax (self, transaction_value):
        return transaction_value * self.TRANSACTION_TAX

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
            self.nstocks = nstocks
            self.balance -= trans_value
            self.taxTransaction(trans_value)
            self.transactions.append(Transaction(nstocks, Transaction.BUY, trans_tax, None, price, None, None))
            if printTransaction:
                self.transactions[-1].print()
        else:
            print('Not enough money.')
        self.state = Simulation.BOUGHT

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
        self.state = Simulation.IDLE
        return treturn

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
    Simulate operation, day by day, in the given position range
    '''
    def simulate (self, t0, t1):
        t = t0

        def next ():
            if t == t1:
                return None, None
            p, d = self.prices[t], self.datetimes[t]
            t += 1
            return p, d

        def history ():
            if t0 > 0:
                return self.prices[0:t0], self.datetimes[0:t0]
            else:
                return None

        return next, history



'''
'''
def save_point (x,xi, y,yi):
    x.append(xi)
    y.append(yi)

def return_percentage (x, of_y):
    diff = x - of_y
    return (diff * 100.0) / of_y

if __name__ == '__main__':

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
    STOP_LOSS =  1.0 * 1e-2
    STOP_GAIN =  1.5 * 1e-2

    IDLE           = 1
    IDLE_WAITING   = 2
    BOUGHT         = 4
    BOUGHT_WAITING = 8
    DONE           = 16
    state          = IDLE

    T        = 390*1
    ndays    = int(float(len(sim.prices)) / float(T))
    balances = []
    returns  = []

    day         = -1
    plot_graphs = False
    t0, t1      = 0, ndays
    if day >= 0:
        t0, t1 = day, day+1

    # For each day in interval [t0,t1)
    for d in range(t0,t1):

        print('\n===============================================================')
        print(f'DAY: {d}')
        print('===============================================================\n')

        day_prices = sim.prices[d*T:(d+1)*T]
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []

        state = IDLE

        for i in range(len(day_prices)):

            price          = day_prices[i]
            sim.net_return = 0.0

            sma, loband, hiband = ts.BB(day_prices[:i+1], winsize=20, stdmult=2.2)
            # ema = ts.EMA(day_prices[:i+1])
            sma2 = ts.SMA(day_prices[:i+1], winsize=50)

            # Forcing sell in the end of the day -------------------------------
            if i == len(day_prices)-1:
                if state == BOUGHT or state == BOUGHT_WAITING:
                    sim.net_return += sim.sell(price, printTransaction=True)
                    save_point(sell_x,i, sell_y,price)
                    state = DONE
                continue
            #-------------------------------------------------------------------

            else:
                # Implementing STOP_LOSS strategy ----------------------------------
                if state == BOUGHT or state == BOUGHT_WAITING:
                    last    = sim.transactions[-1]
                    spent   = (last.nstocks*last.buy_price) + last.tax
                    earning = (sim.nstocks*price) - (sim.nstocks*price * sim.transaction_tax)
                    result  = earning-spent
                    # if reached STOP_LOSS
                    if result < 0 and abs(result) > spent * STOP_LOSS:
                        sim.net_return += sim.sell(price)
                        save_point(sell_x,i, sell_y,price)
                        state = IDLE
                        continue
                #-------------------------------------------------------------------

                # Implementing STOP_GAIN strategy ----------------------------------
                if state == BOUGHT or state == BOUGHT_WAITING:
                    last     = sim.transactions[-1]
                    spent    = (last.nstocks*last.buy_price) + last.tax
                    earning  = (sim.nstocks*price) - (sim.nstocks*price * sim.transaction_tax)
                    result   = earning-spent
                    # If reached STOP_GAIN
                    if result > 0 and result > spent * STOP_GAIN:
                        sim.net_return += sim.sell(price)
                        save_point(sell_x,i, sell_y,price)
                        state = IDLE
                        continue
                #-------------------------------------------------------------------

                import math
                threshold = 0.1 #((1 + math.sqrt(5)) / 2.0) * 1e-1

                # if sma[i] < ema[i]:
                if sma2[i] - sma[i] > threshold and price < loband[i]:
                    if state == IDLE:
                        sim.buy(price, printTransaction=True)
                        save_point(buy_x,i, buy_y,price)
                        state = BOUGHT
                        continue

                # elif ema[i] < sma[i]:
                elif sma[i] - sma2[i] > threshold and price > hiband[i]:
                    if state == BOUGHT:
                        sim.net_return += sim.sell(price, printTransaction=True)
                        save_point(sell_x,i, sell_y,price)
                        state = IDLE
                        continue

        print(f'==> RETURNS: ${sim.net_return}')
        print(f'==> BALANCE: ${sim.balance}')
        returns.append(sim.net_return)
        balances.append(sim.balance)
        if plot_graphs:
            sim.plot_analysis (day_prices, buy_x, buy_y, sell_x, sell_y, sma, ema)

    # Showing balance after simulation finished
    plt.title(label=f'Simulating {sim.file}')
    plt.plot(balances, label='Amount')
    plt.legend()
    plt.show()

    # Showing returns after simulation finished
    plt.title(label=f'Simulating {sim.file}')
    plt.plot(returns, label='Returns')
    # plt.plot(ts.normalize_range(returns,-1,1), label='Returns')
    plt.plot()
    plt.legend()
    plt.show()
