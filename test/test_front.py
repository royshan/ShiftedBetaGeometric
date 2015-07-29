from __future__ import print_function
import pandas as pd
import numpy as np

from tests import make_raw_article_data
import sys
sys.path.append('../shifted_beta_survival/')
from ShiftedBetaSurvival import ShiftedBetaSurvival

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

    print(df)

def article_data():

    data = make_raw_article_data()
    data['category'] = data['category'].apply(lambda x: 'bad' if x == 0 else 'good')
    data['category'] = data['category'].astype('category')

    sbs = ShiftedBetaSurvival(age='age', alive='alive', features='category',
                              gamma=1e-3, verbose=True)
    sbs.fit(data, restarts=5)

    print(sbs.predict_params(data))
    #print(sbs.predict_survival(data, True))

if __name__ == '__main__':

    #test_get_cats()
    article_data()
