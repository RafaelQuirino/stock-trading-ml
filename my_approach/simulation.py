#!/home/rafael/Documents/workspace/github/stock-trading-ml/venv/bin/python3

import sys
import math
import numpy as np
import time_series as ts
import matplotlib.pyplot as plt
from datetime import datetime as dt

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
        self.ttype          = ttype
        self.tax            = tax
        self.buy_datetime   = buy_datetime
        self.buy_price      = buy_price
        self.sell_datetime  = sell_datetime
        self.sell_price     = sell_price

    def output(self, treturn=0.0):
        if self.ttype == Transaction.BUY:
            print(f'--- Bought {self.nstocks} stocks by ${self.buy_price} each with tax ${self.tax} at {self.buy_datetime}')
        elif self.ttype == Transaction.SELL:
            print(f'--- Sold {self.nstocks} stocks by ${self.sell_price} each with tax ${self.tax} at {self.sell_datetime}, return: ${treturn}\n')

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
    Simulation constructor
    '''
    def __init__(self,
        # Data arguments
        prices, datetimes, file,
        # Trading arguments
        balance=0.0, nstocks=0
    ):
        # Data
        self.file      = file
        self.prices    = prices
        self.datetimes = []
        for timestamp in datetimes:
            self.datetimes.append(dt.fromtimestamp(timestamp))
        self.datetimes = np.array(self.datetimes)

        # Correcting data ------------------------------------------------------
        p = [self.prices[0]]
        d = [self.datetimes[0]]
        for i in range(1,len(self.datetimes)):
            if self.datetimes[i] != d[-1]:
                p.append(self.prices[i])
                d.append(self.datetimes[i])
        self.prices    = np.array(p)
        self.datetimes = np.array(d)
        #-----------------------------------------------------------------------

        # Simulation state
        self.t  = 0
        self.t0 = 0
        self.d0 = self.datetimes[0]
        self.d1 = self.datetimes[len(self.datetimes)-1]

        # Trading state
        self.balance         = balance
        self.nstocks         = nstocks
        self.transaction_tax = Simulation.TRANSACTION_TAX
        self.transactions    = []
        self.net_return      = 0.0

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
    def buy (self, price, datetime, nstocks, printTransaction=False):
        if nstocks <= 0:
            print('Can not buy 0 or a negative number of stocks.')
            return False

        # Calculating transaction price and tax ---------
        trans_value  = nstocks*price
        trans_tax    = self.transactionTax(trans_value)
        #------------------------------------------------
        # We can make the transaction
        if self.balance > trans_value + trans_tax:
            # Trading stocks ------------
            self.balance -= trans_value
            self.nstocks += nstocks
            #---------------------------
            # Making the transaction ----------------------------------------------------------------------------
            self.taxTransaction(trans_value) # Taxing the transaction
            self.transactions.append(Transaction(nstocks, Transaction.BUY, trans_tax, datetime, price, None, None))
            if printTransaction:
                self.transactions[-1].output()
            #----------------------------------------------------------------------------------------------------
            return True
        # We can't make the transaction
        else:
            print('Not enough money.')
            return False

    '''
    Sell all stocks currently being held
    '''
    def sell (self, price, datetime, nstocks, printTransaction=False):
        # We can make the transaction
        if nstocks <= self.nstocks:
            # Calculating transaction price and tax ---------
            trans_value   = nstocks*price
            trans_tax     = self.transactionTax(trans_value)
            #------------------------------------------------
            # Trading stocks ------------
            self.nstocks  -= nstocks
            self.balance  += trans_value
            #----------------------------
            # Making the transaction ----------------------------------------------------------------------------
            self.taxTransaction(trans_value) # Taxing the transaction
            last = self.transactions[-1]
            self.transactions.append(Transaction(nstocks, Transaction.SELL, trans_tax, None, None, datetime, price))
            #----------------------------------------------------------------------------------------------------
            # Calculating and saving the return -------------------------------------------
            treturn = (trans_value-trans_tax) - (last.nstocks * last.buy_price + last.tax)
            self.net_return += treturn
            #------------------------------------------------------------------------------
            if printTransaction:
                self.transactions[-1].output(treturn=treturn)
            return treturn
        # We can't make the transaction
        else:
            print('Not enough stocks.')
            return None

    '''
    '''
    def init (self, d0, d1):
        print(f'\nSIMULATING {self.file} FROM {d0} TO {d1}')
        self.t  = 0
        self.d0 = d0
        self.d1 = d1
        for i in range(len(self.datetimes)):
            if self.datetimes[i] >= self.d0:
                self.t = i
                break
        self.t0 = self.t
        return self

    '''
    '''
    def next (self):
        if self.t >= len(self.prices) or self.datetimes[self.t] >= self.d1:
            return None, None
        else:
            p, d = self.prices[self.t], self.datetimes[self.t]

            while d == self.datetimes[self.t-1]:
                self.t += 1
                if self.t >= len(self.prices) or self.datetimes[self.t] >= self.d1:
                    return None, None
                else:
                    p, d = self.prices[self.t], self.datetimes[self.t]

            self.t += 1
            return p, d

    '''
    '''
    def history (self):
        if self.t0 > 0:
            prices = [self.prices[0]]
            datets = [self.datetimes[0]]
            last_d = datets[0]
            for i in range(1,self.t0):
                if self.datetimes[i] != last_d:
                    prices.append(self.prices[i])
                    datets.append(self.datetimes[i])
                    last_d = datets[len(datets)-1]
            prices, datets = np.array(prices), np.array(datets)
            return prices, datets
        else:
            return np.array([]), np.array([])

    '''
    '''
    def seeNext (self):
        if self.t >= len(self.prices) or self.datetimes[self.t] >= self.d1:
            return None, None
        else:
            p, d = self.prices[self.t], self.datetimes[self.t]
            return p, d
    '''
    '''
    def isEndOfDay (self):
        if self.t >= len(self.prices) or self.datetimes[self.t] >= self.d1:
            return True
        currdatetime = self.datetimes[self.t-1]
        nextdatetime = self.datetimes[self.t]
        if nextdatetime.day != currdatetime.day:
            return True
        else:
            return False
