from __future__ import print_function
import pandas as pd
import numpy as np

import sys
sys.path.append('../shifted_beta_survival/')
from DataHandler import DataHandler


def get_data(nrows=10000):
    df = pd.read_csv('../data/data_handler.csv', nrows=nrows)
    df['origin'] = df['origin'].astype('category')
    df['frequency'] = df['frequency'].astype('category')
    df['profession'] = df['profession'].astype('category')
    return df

def test_get_cats():

    df = get_data(nrows=500)
    te = get_data()

    dh = DataHandler(age='age',
                     alive='alive',
                     features=['origin', 'profession', 'frequency', 'amount'],
                     #features=['amount'],
                     bias=False
                     )

    dh.fit(df)
    dh.transform(te)

if __name__ == '__main__':

    test_get_cats()

