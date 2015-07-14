from __future__ import print_function
from DataHandler import DataHandler
from ShiftedBeta import ShiftedBeta
import numpy
import pandas
from math import exp
from scipy.special import hyp2f1


class ShiftedBetaSurvival(object):
    """
    what is this?
    """

    def __init__(self, age,
                 alive,
                 features=None,
                 gamma=1.0,
                 gamma_beta=1.0,
                 bias=True,
                 verbose=False):
        """
        stuff ...

        :param cohort:
        :param age:
        :param category:
        :param gamma:
        :param verbose:
        :return:
        """
        # ORIGINAL DATA
        # The original dataset is stored in the following two variables.
        #   df: Original data in its original format
        # data: Original data transformed into map of category-value -
        #       (cohort population, population lost) pairs.
        self.df = None
        self.data = None

        # COLUMN'S NAMES
        # The names of the columns used as cohort, age and category throughout
        # the code.
        self.age = age
        self.alive = alive

        # whether or not we should add a bias
        self.bias = bias

        # If the category name was passed as a single string, we turn it into
        # a list of one element (not list of characters, as you would get with
        # list('abc').
        if isinstance(features, str):
            features = [features]
        # Try to explicitly transform category to a list (perhaps it was passed
        # as a tuple or something. If it was None to begin with, we catch a
        # TypeError and move on.
        try:
            self.features = sorted(features)
        except TypeError:
            self.features = None
            # Set bias to true if no features are being used
            self.bias = True

        if bias:
            if self.features is None:
                self.names = ['bias']
            else:
                self.names = ['bias'] + self.features
        else:
            self.names = self.features

        # SHIFTED-BETA-GEOMETRIC MODEL
        # Since the training data is only available at training time (upon
        # calling .fit(), we must postpone the initialization of the
        # ShiftedBEta object, so we simple initialize it as None.
        self.sb = None
        self.sb_params = None

        # Instance parameters used to hold the post-train values of alpha and
        # beta.
        self.alpha = None
        self.beta = None

        # L2 regularizer
        # The L2 regularization is governed by the size of the parameter gamma.
        # In the current implementation a value of zero for the regularization
        # constant is allowed, since the biar of the linear model is not
        # regularized, and a pathological solution to the optimization problem
        # is unlikely to happen.
        # However, negatives values for gamma not only don't make sense, but
        # they break everything. So we make sure the value of gamma passed is
        # a resonable one.#
        if gamma < 0:
            raise ValueError("The regularization constant gamma must be a "
                             "non-negative real number. A negative value of"
                             " {} was passed.".format(gamma))
        self.gamma = gamma

        if gamma_beta is None:
            # Use gamma value by default
            self.gamma_beta = gamma
        else:
            # make sure is not zero
            if gamma_beta < 0:
                raise ValueError("The regularization constant gamma must be a "
                                 "non-negative real number. A negative value of"
                                 " {} was passed.".format(gamma))

            self.gamma_beta = gamma_beta

        # trained?
        self.trained = False

        # verbose controller
        self.verbose = verbose

        # Create objects!
        # DATA-HANDLER OBJECT
        # The DataHandler object may be created without the training data, so
        # we do it here.
        #self.dh = DataHandler()

        # Shifted beta model object
        # create shifted beta object
        self.sb = ShiftedBeta(gamma_alpha=self.gamma,
                              gamma_beta=self.gamma_beta,
                              add_bias=self.bias,
                              verbose=self.verbose)

    def fit(self, df, restarts=1):
        """
        A fit method to train the model.

        :param df: pandas DataFrame
            A pandas DataFrame with similar schema as the one used to train
            the model. Similar in the sense that the columns used as cohort,
            age and categories must match. Extra columns with not affect
            anything.

        :param restarts: int
            Number of times to restart the optimization procedure with a
            different seed, to avoid getting stuck on local maxima.
        """

        if self.features is None:
            x = None
        else:
            x = df[self.features].values
        # targets
        y = df[self.age].values.astype(int)
        z = df[self.alive].values.astype(int)

        # fit to data!
        self.sb.fit(age=y,
                    alive=z,
                    X=x,
                    restarts=restarts)

        alpha, beta = self.sb.get_params()
        self.sb_params = dict(alpha=dict(zip(self.names, alpha)),
                              beta=dict(zip(self.names, beta)))

        # Trained successful means training is done!
        self.trained = True

    def summary(self):
        """

        :return:
        """

        # Model ran?
        if not self.trained:
            raise RuntimeError('Train the model first!')

        out = pandas.DataFrame(self.sb_params)

        # Take exponential
        out['exp(alpha)'] = numpy.exp(out['alpha'])
        out['exp(beta)'] = numpy.exp(out['beta'])

        return out

    def get_coeffs(self):

        if not self.trained:
            raise RuntimeError('Train the model first!')

        return self.sb.get_coeffs()

    def _predict_coefficients(self, X):
        return self.sb.predict(X)

    def predict_ltv(self, df, key=None, **kwargs):
        """

        :param df:
        :param kwargs:
        :return:
        """
        if self.features is None:
            x = None
        else:
            x = df[self.features].values

        params = self._predict_coefficients(x)

        ltvs = self.derl(params[:, 0], params[:, 1], renewals=df[self.age].values, **kwargs)

        if key is None:
            return pandas.DataFrame(data=ltvs, columns=['ltv'])
        else:
            out = df[[key]].copy()
            out['ltv'] = ltvs
            return out

    def churn_p_of_t(self, n_periods=12):
        """
        churn_p_of_t computes the churn as a function of time curve. Using
        equation 7 from [1] and the alpha and beta coefficients obtained by
        training this model, it computes P(T = t) recursively, returning either
        the expected value or an array of values.


        :param n_periods: Int
            The number of months to compute the curve for

        :return: Float or ndarray
            Returns as float if expected is true and a ndarray is expected is
            false.
        """
        # Spot checks making sure the values passed make sense!
        if n_periods < 0 or not isinstance(n_periods, int):
            raise ValueError("The number of periods must be a non-zero "
                             "integer")

        if not self.trained:
            raise RuntimeError('Train the model first!')

        churn_by_cate = {}

        for category, val_dict in self.sb.get_coeffs().items():

            # add category dict
            churn_by_cate[category] = {}

            for value, param in val_dict.items():

                # Load alpha and beta sampled from the posterior. These fully
                # determine the beta distribution governing the customer level
                # churn rates
                alpha = param['alpha']
                beta = param['beta']

                # --- Initialize Output ---
                # Initialize the output as a matrix of zeros. The number of rows is
                # given by the total number of samples, while the number of columns
                # is the number of months passed as a parameter.
                churn_by_cate[category][value] = numpy.zeros(n_periods)

                # --- Fill output recursively (see eq.7 in [1])---

                # Start with month one (churn rate of month zero was set to 0 by
                # definition).
                churn_by_cate[category][value][1] = alpha / (alpha + beta)

                # Calculate remaining months recursively using the formulas
                # provided in the original paper.
                for i in range(2, n_periods):

                    month = i
                    update = (beta + month - 2) / (alpha + beta + month - 1)

                    # None that i + 1 is simply the previous value, since val
                    # starts with the third entry in the array, but I starts
                    # counting form zero!
                    churn_by_cate[category][value][i] += update * churn_by_cate[category][value][i - 1]

        # Mesh category name and values together to output in a DataFrame format.
        out = pandas.DataFrame()
        for category, value_dict in churn_by_cate.items():
            for value, array in value_dict.items():
                out[category + '_' + str(value)] = array

        return out

    def survival_function(self, n_periods=12, renewals=0):
        """
        survival_function computes the survival curve obtained from the model's
        parameters and assumptions. Using equation 7 from [1] and the alpha and
        beta coefficients obtained by training this model, it computes S(T = t)
        recursively, returning either the expected value or an array of values.
        To do so it must first invoke the self.churn_p_of_t method to calculate
        the monthly churn rates for the given time window, and then use it to
        compute the survival curve recursively.

        :param n_periods: Int
            The number of months to compute the curve for

        :return: Float or ndarray
            Returns as float if expected is true and a ndarray is expected is
            false.
        """
        # Spot checks making sure the values passed make sense!
        if renewals < 0 or not isinstance(renewals, int):
            raise ValueError("The number of renewals must be a non-zero "
                             "integer")

        if n_periods < 0 or not isinstance(n_periods, int):
            raise ValueError("The number of periods must be a non-zero "
                             "integer")

        if not self.trained:
            raise RuntimeError('Train the model first!')

        # --- Churn Rates ---
        # Start by calling the method churn_p_of_t to calculate the monthly
        # churn rates gives the model's fitted parameters and the parameters
        # passed to the function.
        # Since this method is able to return the normalized tail of the
        # survival function (the retention rates using other than the
        # starting point as normalization), we must make sure we compute the
        # churn rates for a long enough period. Therefore we pass the value
        # n_months + renewals to the n_month parameter of the churn_p_of_t
        # which guarantees the churn rate curve extends far enough into the
        # future.#
        p_of_t = self.churn_p_of_t(n_periods=n_periods + renewals)

        surv_by_cate = {}

        for category, val_dict in self.sb.get_coeffs().items():

            # add category dict
            surv_by_cate[category] = {}

            for value, param in val_dict.items():

                # Dataframe name
                name = category + '_' + str(value)

                # --- Initialize output ---
                # The output is initialized as a zero matrix with the same shape
                # as the churn rates matrix
                surv_by_cate[category][value] = numpy.zeros(p_of_t.shape[0])

                # The initial value is one by definition (in this model death at
                # t=0 is no considered).
                surv_by_cate[category][value][0] = 1

                # The value of month is simply given by the naive formula
                #       1 - churn(t=1)
                surv_by_cate[category][value][1] = 1 - p_of_t[name].values[1]

                # The remaining values are calculated recursively using eq. 7 [1].
                for i, val in enumerate(p_of_t[name].values[2:]):

                    # Something here...
                    surv_by_cate[category][value][i + 2] = surv_by_cate[category][value][i + 1] - val

        # To data-frame and some re-formatting
        out = pandas.DataFrame()
        for category, value_dict in surv_by_cate.items():
            for value, array in value_dict.items():
                out[category + '_' + str(value)] = array
        out = out.iloc[renewals:]
        out.index = range(out.shape[0])

        return out

    @staticmethod
    def derl(alpha, beta, arpu=1.0, discount_rate=0.005, renewals=0):
        """
        Discounted Expected Residual Lifetime, as derived in [2].
        See equation (6).

        :param alpha: float
            Value of sBG alpha param

        :param beta: float
            Value of sBG beta param

        :param arpu: Float
            Average Revenue Per User

        :param discount_rate: Float
            discount rate

        :param renewals: Int
            customer's contract period (customer has made n-1 renewals)

        :return: Float
            The DERL for the above values.
        """

        # In the off change alpha and beta are not numpy array we make sure
        # they are.
        alpha = numpy.asarray(alpha)
        beta = numpy.asarray(beta)

        # To make it so that the formula resembles that of the paper we define
        # the parameter n as below.
        n = renewals + 1

        # The equation is two long, so we break in two parts.
        f1 = (beta + n - 1) / (alpha + beta + n - 1)
        f2 = hyp2f1(1., beta + n, alpha + beta + n, 1. / (1. + discount_rate))

        return arpu * f1 * f2
