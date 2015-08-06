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
    df['lamount'] = np.log(df.amount + 1)
    return df

def test_get_cats():

    df = get_data(nrows=100)
    te = get_data(nrows=250)

    dh = DataHandler(age='age',
                     alive='alive',
                     #features=['origin', 'profession', 'frequency', 'amount', 'lamount', 'is_contractor'],
                     #features=['amount', 'lamount'],
                     features=None,
                     bias=False,
                     normalize=True
                     )

    dh.fit(df)
    dh.transform(df)
    x, y, z = dh.transform(te[['origin', 'profession', 'frequency', 'amount', 'lamount', 'is_contractor', 'age']])
    #x, y, z = dh.transform(te)

    print(x, y, z)
    print(dh.stats)
    print(dh.feature_map)

    print(dh.get_names())

if __name__ == '__main__':

    test_get_cats()

