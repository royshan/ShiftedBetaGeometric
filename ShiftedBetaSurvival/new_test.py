from __future__ import print_function
import pandas
import numpy

from DataHandler import DataHandler
from ShiftedBeta import ShiftedBeta
from ShiftedBetaSurvival import ShiftedBetaSurvival

xraw = numpy.asarray([[1, 0, 1],
                      [1, 1, 1],
                      [1, 0, 1],
                      [1, 0, 0],
                      [1, 1, 0]])

yraw = numpy.asarray([3, 2, 4, 3, 1])
zraw = numpy.asarray([1, 1, 0, 0, 1])


def make_raw_article_data():

    highend_lost = numpy.asarray([0, 131, 257, 347, 407, 449, 483, 509])
    regular_lost = numpy.asarray([0, 369, 532, 618, 674, 711, 738, 759])

    # high end guys
    data_he = numpy.zeros((1000, 5), dtype=int)
    data_he[:, -2] = 8
    data_he[:, 0] = numpy.arange(data_he.shape[0])

    for age in reversed(highend_lost):
        data_he[:age, -2] -= 1

    # regular end guys
    data_re = numpy.zeros((1000, 5), dtype=int)
    data_re[:, -2] = 8
    data_re[:, 0] = numpy.arange(data_re.shape[0], 2 * data_re.shape[0])
    data_re[:, 2] = 1

    for age in reversed(regular_lost):
        data_re[:age, -2] -= 1

    data_he[:, -1][data_he[:, -2] == 8] = 1
    data_re[:, -1][data_re[:, -2] == 8] = 1

    out_data = numpy.concatenate((data_he, data_re), axis=0)

    data = pandas.DataFrame(data=out_data, columns=['id', 'cohort', 'category', 'age', 'alive'])

    return data


def sb_test(x, y, z):

    sb = ShiftedBeta(verbose=True, gamma=1e-8)

    #wa = numpy.asarray([-0.40346710544549125, 0.05249018262139654])
    #wb = numpy.asarray([1.3365787688739577, -1.1693708498900512])

    sb.fit(y, z, x, restarts=2)
    print(sb.opt, numpy.exp(sb.opt), numpy.exp(sb.alpha[0]) / (numpy.exp(sb.alpha[0]) +
                                                       numpy.exp(sb.beta[0])))


def sb_test2():

    data = pandas.read_csv('../data/new_data.csv', nrows=100000)

    print(data.head())

    #x = data.values[:, [0, 2, 3, 4, 5, 6, 7]]
    x = data.values[:, :-3]
    y = data.values[:, -2].astype(int)
    z = data.values[:, -1].astype(int)

    sb = ShiftedBeta(verbose=True, gamma=1e2)

    sb.fit(y, z, x, restarts=1)
    print(sb.alpha, sb.beta, numpy.exp(sb.alpha[0]) / (numpy.exp(sb.alpha[0]) +
                                                       numpy.exp(sb.beta[0])))
    #print(sb.opt, numpy.exp(sb.opt))



if __name__ == '__main__':

    # sb_test(xraw, yraw, zraw)
    #data = make_raw_article_data().iloc[:]
    #sb_test(data[['category']], data['age'], data['alive'])

    sb_test2()

    #print(make_raw_article_data())
