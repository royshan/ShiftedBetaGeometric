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

    sb = ShiftedBeta(verbose=True, gamma=1e-10)

    #wa = numpy.asarray([-0.40346710544549125, 0.05249018262139654])
    #wb = numpy.asarray([1.3365787688739577, -1.1693708498900512])
    wa = numpy.asarray([-0.40346710544549125])
    wb = numpy.asarray([1.333])

    #print(sb._compute_alpha_beta(x, wa, wb))
    #print(sb._logp(numpy.concatenate((numpy.ones((x.shape[0], 1)), x), axis=1), y, z, wa, wb))
    #print(sb._logp(numpy.ones((y.shape[0], 1)), y, z, wa, wb))

    sb.fit(y, z, restarts=2)
    #print(sb.opt, numpy.exp(sb.opt))



if __name__ == '__main__':
    # sb_test(xraw, yraw, zraw)

    data = make_raw_article_data().iloc[:1000]
    sb_test(data[['category']], data['age'], data['alive'])

    #print(make_raw_article_data())
