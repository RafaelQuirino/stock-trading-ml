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
    prices, datetimes, file = ts.import_data('data/abev3_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/bbas3_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/itub4_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/jbss3_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/petr4_2019.csv', 'preco', 'hora', 0)
    # prices, datetimes, file = ts.import_data('data/petr4_2020.csv', 'preco', 'hora', 0)


    #===========================================================================
    # Testing Simulation
    #===========================================================================
    d0 = dt(2019,1,3)
    d1 = dt(2020,1,1)
    initial_balance = 1000.0
    sim = s.Simulation(prices, datetimes, file, balance=initial_balance).init(d0,d1)
    bot = r.Robot(balance=initial_balance).setHistory(sim.history()).init()

    curr_day   = -1
    curr_month = -1
    p, d = sim.next()
    while (p,d) != (None,None):

        # Logging new day and new month ----------------------------------------
        day, month = d.day, d.month
        if month != curr_month:
            print('\n=========================================================')
            print('NEW MONTH: ', d)
            print('=========================================================\n')
            curr_month = month
        if day != curr_day:
            print('\n------------------------------------')
            print('New day: ', d)
            print('------------------------------------\n')
            curr_day = day
        #-----------------------------------------------------------------------

        # Robot in action ------------------------------------------------------
        if sim.isEndOfDay():
            if bot.state == r.Robot.STATE_BOUGHT or bot.state == r.Robot.STATE_BOUGHT_WAITING:
                print('>>> FORCED SELL')
                bot.sell(p, d, sim, printTransaction=True)
        else:
            action = bot.query(p,d)
            if action == r.Robot.ACTION_BUY:
                bot.buy(p, d, sim, printTransaction=True)
            elif action == r.Robot.ACTION_SELL:
                bot.sell(p, d, sim, printTransaction=True)
        #-----------------------------------------------------------------------

        # Get next values, log day and month results ---------------------------
        newp, newd = sim.next()

        if (newp,newd) != (None,None):
            if newd.day != d.day:
                # Report Day
                print(f'\n    => BALANCE AT END OF DAY:  ${bot.balance}\n')
            if newd.month != d.month:
                # Report Month
                print(f'\n    => BALANCE AT END OF MONTH:  ${bot.balance}\n')

        p, d = newp, newd
        #-----------------------------------------------------------------------
    print(f'\n=> FINAL BALANCE:  ${bot.balance}\n')
    #===========================================================================
