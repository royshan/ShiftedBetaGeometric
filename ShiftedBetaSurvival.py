import pandas
import numpy
import scipy

from ShiftedBeta import ShiftedBeta
from DataHandler import DataHandler

class ShiftedBetaSurvival(object):

    def __init__(self):

        self.sb = None # ShiftedBeta()
        self.dh = None # DataHandler()

        self.data = None
        self.cohort = None
        self.age = None
        self.category = None

    def fit(self, df, cohort, age, category=None):

        self.df = df
        self.cohort = cohort
        self.age = age
        self.category = category

        self.dh = DataHandler(data=self.df,
                              cohort=self.cohort,
                              age=self.age,
                              category=self.category)

    def summary(self):
        pass

    def get_params(self):
        pass

    def churn(self):
        pass

    def survival_curve(self):
        pass

    def ltv(self):
        pass

    def capped_ltv(self):
        pass

