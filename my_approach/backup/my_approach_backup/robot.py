import simulation as sim
import time_series as ts

class Robot:

    ACTION_BUY  = 0
    ACTION_SELL = 1
    ACTION_WAIT = 2

    def __init__(self, prices=[], datetimes=[]):
        self.prices    = prices
        self.datetimes = datetimes

    def inputPrice (self, newprice, newdatetime):
        self.prices.append(newprice)
        self.datetimes.append(newdatetime)

    def query (self):
        return Robot.ACTION_WAIT
