from __future__ import print_function
import numpy as np
import pandas as pd


def make_raw_article_data():
    """
    A function to generate the customer level data used in [1]

    :return: pandas DataFrame
    """

    # List of number of customers lost per year per quality level
    highend_lost = np.asarray([0, 131, 257, 347, 407, 449, 483, 509])
    regular_lost = np.asarray([0, 369, 532, 618, 674, 711, 738, 759])

    # high-end
    data_he = np.zeros((1000, 5), dtype=float)

    # Start age column with max-age, we will change this later
    data_he[:, -2] = 8

    # is high end indicator
    data_he[:, 1] = 1

    for age in reversed(highend_lost):
        data_he[:age, -2] -= 1

    # regulars
    data_re = np.zeros((1000, 5), dtype=float)
    data_re[:, -2] = 8

    for age in reversed(regular_lost):
        data_re[:age, -2] -= 1

    # those with age 8 are still alive (here we assume they all belong to the
    # same cohort)
    data_he[:, -1][data_he[:, -2] == 8] = 1
    data_re[:, -1][data_re[:, -2] == 8] = 1

    out_data = np.concatenate((data_he, data_re), axis=0)
    np.random.shuffle(out_data)

    # ids
    out_data[:, 0] = np.arange(out_data.shape[0]) + 1000

    # random field
    out_data[:, 2] = np.random.randn(out_data.shape[0])

    data = pd.DataFrame(data=out_data, columns=['id', 'is_high_end', 'random', 'age', 'alive'])

    return data


def generate_model_data():
    return 0


if __name__ == '__main__':
    print(make_raw_article_data().head())
