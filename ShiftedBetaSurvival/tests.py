import pandas
import numpy

from DataHandler import DataHandler
from ShiftedBeta import ShiftedBeta
from ShiftedBetaSurvival import ShiftedBetaSurvival

data_raw = [[1, 0, 5],
            [1, 0, 3],
            [1, 0, 1],
            [1, 0, 2],
            [1, 0, 3],
            [1, 1, 6],
            [1, 0, 4],
            [1, 0, 2],
            [1, 1, 2],
            [1, 1, 6],
            [1, 0, 1.2],
            [2, 1, 1],
            [2, 0, 2],
            [2, 0, 3],
            [2, 1, 1],
            [2, 0, 2],
            [2, 1, 3],
            [2, 0, 5],
            [2, 1, 4],
            [2, 0, 4],
            ]


def make_raw_article_data():

    highend_lost = numpy.asarray([0, 131, 257, 347, 407, 449, 483, 509])
    regular_lost = numpy.asarray([0, 369, 532, 618, 674, 711, 738, 759])

    # high end guys
    data_he = numpy.zeros((1000, 4), dtype=int)
    data_he[:, -1] = 8
    data_he[:, 0] = numpy.arange(data_he.shape[0])

    for age in reversed(highend_lost):
        data_he[:age, -1] -= 1

    # regular end guys
    data_re = numpy.zeros((1000, 4), dtype=int)
    data_re[:, -1] = 8
    data_re[:, 0] = numpy.arange(data_re.shape[0], 2 * data_re.shape[0])
    data_re[:, 2] = 1

    for age in reversed(regular_lost):
        data_re[:age, -1] -= 1

    out_data = numpy.concatenate((data_he, data_re), axis=0)

    data = pandas.DataFrame(data=out_data, columns=['id', 'cohort', 'category', 'age'])

    return data


def format_article_data(data):
    dh = DataHandler(data, 'cohort', 'age', ['category'])

    #print dh.aggregate()
    #print dh.n_lost(dh.aggregate())
    #print dh.paired_data()
    return dh.paired_data()

def init_shifted_beta(data):
    sb = ShiftedBeta(data, verbose=True)

    print sb.categories, sb.n_cats
    #print sb.imap
    #print sb.data
    sb.fit(restarts=3)
    print sb.categories
    print sb.get_coeffs()
    print sb.get_params()

def run_paper_tests():
    paper = make_raw_article_data()
    pairs = format_article_data(paper)

    sb = ShiftedBeta(pairs)
    sb.fit()

    print sb.get_coeffs()
    print sb.get_params()

def test_big_class():
    paper = make_raw_article_data()
    # print paper.head()

    sbv = ShiftedBetaSurvival('cohort', 'age', category='category')
    sbv.fit(paper)

    print sbv.summary()

    print sbv.ltv()

    print
    print sbv.churn_p_of_t()
    print sbv.survival_function(renewals=1)


def format_data_test(data_raw):

    data = pandas.DataFrame(data=data_raw, columns=['cohort', 'kind', 'age'])
    dh = DataHandler(data, 'cohort', 'age', ['kind'])
    print dh.aggregate()
    print dh.n_lost(dh.aggregate())

def big_real_data():

    data = pandas.read_csv('./data/data.csv', nrows=2500)

    sbv = ShiftedBetaSurvival(cohort='cohort', age='age', category=['il1', 'frequency'],
                              gamma=10, verbose=True)
    sbv.fit(data, restarts=2)

    print sbv.summary()
    print

    print data.iloc[:20]
    print sbv.predict_ltv(data.iloc[:20], arpu=20)
    # print data.iloc[:20].apply(lambda row: sbv._predict_coefficients(row), axis=1)

    print sbv._coefficients_combination()

if __name__ == '__main__':

    # data = format_article_data(make_raw_article_data())
    # init_shifted_beta(data)
    # run_paper_tests()
    # format_data_test(data_raw)
    # test_big_class()
    big_real_data()

