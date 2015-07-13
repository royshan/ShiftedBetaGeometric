from __future__ import print_function
from datetime import datetime
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

def print_stats(am, bm, index, names):

    a = 0.0
    b = 0.0
    pname = ''
    for i in index:
        a += am[i]
        b += bm[i]

        pname += names[i]
        pname += '_'

    a = numpy.exp(a)
    b = numpy.exp(b)
    print('{0:15} | {1:5.5}, {2:5.5}, {3:5.5}'.format(pname, a, b, a / (a + b)))


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


def sb_test(x, y, z, names):

    sb = ShiftedBeta(verbose=True, gamma=1e2, gamma_ratio=1.)

    #wa = numpy.asarray([-0.40346710544549125, 0.05249018262139654])
    #wb = numpy.asarray([1.3365787688739577, -1.1693708498900512])

    sb.fit(y, z, x, restarts=2)

    print(sb.alpha, numpy.exp(sb.alpha[0]), numpy.exp(sb.alpha.sum()))
    print(sb.beta, numpy.exp(sb.beta[0]), numpy.exp(sb.beta.sum()))
    print(numpy.exp(sb.alpha[0]) / (numpy.exp(sb.alpha[0]) + numpy.exp(sb.beta[0])))
    print(numpy.exp(sb.alpha[0] + sb.alpha[1]) /
          (numpy.exp(sb.alpha[0] + sb.alpha[1]) + numpy.exp(sb.beta[0] + sb.beta[1])))

    print_stats(sb.alpha, sb.beta, [0], names)
    for i in range(1, x.shape[1] + 1):
        print_stats(sb.alpha, sb.beta, [0, i], names)


def sb_test2():

    data = pandas.read_csv('../data/new_data.csv')
    total_size = data.shape[0]#10000
    index = numpy.arange(data.shape[0])
    numpy.random.shuffle(index)
    index = index[:total_size]

    data = data.iloc[index]
    data.index = numpy.arange(total_size)

    print(data.head())

    names = ['monthly', 'annual']
    #names = list(data.keys())[:-2]

    x = data[names].values
    #x = data.values[:, :-3]
    y = data.values[:, -2].astype(int)
    z = data.values[:, -1].astype(int)

    names = ['bias'] + names

    print("Average age: {0:2} | Average still alive: {1:0.3}".format(y.mean(), z.mean()))

    sb = ShiftedBeta(verbose=True, gamma=5e2, gamma_ratio=0.1)

    sb.fit(y, z, x, restarts=1)
    print(sb.alpha, numpy.exp(sb.alpha[0]), numpy.exp(sb.alpha.sum()))
    print(sb.beta, numpy.exp(sb.beta[0]), numpy.exp(sb.beta.sum()))

    print_stats(sb.alpha, sb.beta, [0], names)
    for i in range(1, x.shape[1] + 1):
        print_stats(sb.alpha, sb.beta, [0, i], names)


if __name__ == '__main__':

    start = datetime.now()

    # sb_test(xraw, yraw, zraw)
    #data = make_raw_article_data().iloc[:]
    #data.insert(1, 'random', numpy.random.randint(0, 2, (data.shape[0], 1)))
    #names = ['bias', 'category', 'random']
    #sb_test(data[['category', 'random']], data['age'], data['alive'], names)

    sb_test2()

    print("main took: {}".format(datetime.now() - start))
