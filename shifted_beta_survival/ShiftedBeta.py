from __future__ import print_function
from scipy.optimize import minimize
from math import log10
from datetime import datetime
import numpy
from scipy.special import hyp2f1

class ShiftedBeta(object):
    """
    This class implements the Shifted-Beta model by P. Fader and B. Hardie,
    however, unlike the original paper, we take the bayesian route and compute
    directly the distributions of parameters alpha and beta using MCMC. These,
    in turn, are used to estimate the expected values of tenure and LTV.

    This model works by assuming a constant in time, beta distributed
    individual probability of churn. Due to the heterogeneity of a cohort's
    churn rates (since each individual will have a different probability of
    churning), expected behaviours such as the decrease of cohort churn rate
    over time arise naturally.

    To train the model we need time evolution of a cohort's population in the
    form:
        c1 = [N_0, N_1, ...]

    Since we have multiple cohorts coexisting at any given month we may
    leverage all this information to train the model.
        c1 = [N1_0, N1_1, ...]
        c2 = [N2_0, N2_1, ...]
        ...
        data = [c1, c2, ...]
    """

    def __init__(self,
                 gamma_alpha=1.0,
                 gamma_beta=None,
                 add_bias=True,
                 verbose=False):

        self.alpha = None
        self.beta = None

        # regularizer
        if gamma_alpha < 0:
            raise ValueError("The regularization constant gamma must be a "
                             "non-negative real number. A negative value of"
                             " {} was passed.".format(gamma_alpha))

        # different regularization parameters for alpha and beta can be helpful
        # so we include a ratio parameters that allows them to be set
        # differently
        self.gammaa = gamma_alpha
        if gamma_beta is None:
            self.gammab = self.gammaa
        else:
            self.gammab = gamma_beta

        # bias
        self.bias = add_bias

        # ops obj
        self.opt = None

        # verbose param
        self.verbose = verbose

        # size of dataset
        self.n_samples = 0

    @staticmethod
    def _recursive_retention_stats(alpha, beta, num_periods):
        """
        A function to calculate the expected probabilities recursively.
        Using equation 7 from [1] and the alpha and beta coefficients
        obtained by training this model, it computes P(T = t) recursively,
        returning a list with all values.

        Survival function recursive calculation. Using equation 7 from [1]
        and the alpha and beta coefficients obtained by training this
        model, it computes S(T = t) recursively, returning a list of all
        computed values.. To do so it must first invoke the function
        P_T_is_t calculate the monthly churn rates for the given time
        window, and then use it to compute the survival curve recursively.

        :param alpha: float
            The distribution for the alpha parameter.

        :param beta: float
            The distribution for the beta parameter.

        :param num_periods: Int
            The number of periods for which the probability of churning
            should be computed.

        :return: (list, list)
            A list with probability of churning for all periods from month
            zero to num_periods.
        """
        alpha = max(min(alpha, 1e4), 1e-4)
        beta = max(min(beta, 1e4), 1e-4)

        # Initialize list with t = 0 and t = 1 values
        p_old = None
        s_old = 1

        p_new = alpha / (alpha + beta)
        s_new = 1. - p_new

        for t in range(2, num_periods + 1):

            p_old = 1. * p_new
            s_old = 1. * s_new

            # Compute latest p value and append
            p_new = (beta + t - 2.) / (alpha + beta + t - 1.) * p_old

            # use the most recent appended p value to keep building s
            s_new = s_old - p_new

        # finish this...
        return p_new, s_old

    @staticmethod
    def _compute_alpha_beta(X, alpha, beta):
        """

        :param X:
        :param alpha:
        :param beta:
        :return:
        """

        alpha_weights = (alpha * X).sum(axis=1)
        beta_weights = (beta * X).sum(axis=1)

        return numpy.exp(alpha_weights), numpy.exp(beta_weights)

    def _logp(self, X, age, alive, wa, wb):
        """
        The LogLikelihood function. Given the data and relevant
        variables this function computed the loglikelihood.

        :param X:
        :param age:
        :param alive:
        :param wa:
        :param wb:
        :return:
        """
        # --- LogLikelihood (One Cohort at a Time) --- #
        # We calculate the LogLikelihood for each cohort separately and
        # combining them. From appendix B in [1] it is easy to see that
        # the extension of the model to multiple cohorts of different
        # sizes is simply given a similar product as in B1, except that
        # each month of each cohort will contribute with a term like:
        #       P(T = t | alpha, beta) ** n_t
        # Which, when taking the log, translates to a sum similar to B3,
        # but extended to include all cohorts.
        log_like = 0.0

        # L2 regularizer
        # something about l2 regularization
        # Explain why the intercept (zero-th index portion of alpha and beta)
        # are not subject to regularization. Also, think whether this is the
        # best way of handling this, or whether adding a dedicated intercept
        # is a better choice.
        l2_reg = self.gammaa * sum(wa[1:]**2) + self.gammab * sum(wb[1:]**2)

        # update ll with regularization val.
        log_like -= l2_reg

        # get real alpha and beta
        alpha, beta = self._compute_alpha_beta(X, wa, wb)

        # loop over data doing stuff
        for y, z, a, b in zip(age, alive, alpha, beta):

            # add contribution of current customer to likelihood
            log_like += numpy.log(self._recursive_retention_stats(a, b, y)[z])

        # Negative log_like since we will use scipy's minimize object.
        return -log_like

    def fit(self, age, alive, X=None, restarts=50):

        # Free space when printing
        print_space = max(int(log10(restarts)) + 1, 5)

        # store the number of samples we are dealing with
        self.n_samples = age.shape[0]

        # Add bias
        if X is None:
            X = numpy.ones((self.n_samples, 1))
        elif self.bias:
            X = numpy.concatenate((numpy.ones((self.n_samples, 1)), X), axis=1)

        # Now we have X we can calculate the number of params we need
        n_params = X.shape[1]

        # guesses of initial parameters
        initial_guesses = 0.1 * numpy.random.randn(restarts, 2 * n_params) - 0.01

        # Initialize optimal value to None
        # I choose not to set it to, say, zero, or any other number, since I am
        # not sure that the log-likelihood is bounded in anyway. So is better to
        # initialize with None and use the first optimal value start the ball
        # rolling.
        optimal = None

        # clock
        start = datetime.now()

        if self.verbose:
            print('Starting Optimization with parameters:')
            print('{:>15}: {}'.format('Samples', self.n_samples))
            print('{:>15}: {}'.format('gamma (alpha)', self.gammaa))
            print('{:>15}: {}'.format('gamma (beta)', self.gammaa))
            print('{:>15}: {}'.format('bias', self.bias))
            print('{:>15}: {}'.format('Seeds', restarts))

            print()
            print("{0:^{3}} | {1:^10} | {2:13} |".format('Step',
                                                         'Time',
                                                         'LogLikelihood',
                                                         print_space))
            print("-"*35)

        # Run likelihood optimization for several steps...
        # noinspection PyTypeChecker
        for step, guess in enumerate(initial_guesses):

            # --- Optimization
            # something...
            new_opt = minimize(lambda p: self._logp(X=X,
                                                    age=age,
                                                    alive=alive,
                                                    wa=p[:n_params],
                                                    wb=p[n_params:]),
                               guess,
                               bounds=[(None, None)] * n_params * 2
                               )

            # If first run...
            if optimal is None:
                optimal = new_opt.fun
                self.opt = new_opt.x

            # Have we found a better value yet?
            if new_opt.fun > optimal:
                optimal = new_opt.fun
                self.opt = new_opt.x

            if self.verbose:
                print("{0: {3}} | {1:^10.10} | {2:13.8} |".format(step + 1,
                                                                  datetime.now() - start,
                                                                  optimal,
                                                                  print_space))

        self.alpha = self.opt[:n_params]
        self.beta = self.opt[n_params:]

    def predict(self, X=None):
        """
        computes alpha and beta for a each row of a given matrix x with the
        same shape as the data the model was trained on, obviously.

        :param x:
        :return:
        """
        #  use this without bias
        if X is None:
            alpha = numpy.exp(self.alpha[0]) * numpy.ones(self.n_samples)
            beta = numpy.exp(self.beta[0]) * numpy.ones(self.n_samples)
        else:
            alpha, beta = self._compute_alpha_beta(X,
                                                   self.alpha[1:],
                                                   self.beta[1:])

            # add bias contribution
            alpha *= numpy.exp(self.alpha[0])
            beta *= numpy.exp(self.beta[0])

        return numpy.vstack([alpha, beta]).T

    def get_params(self):
        return self.alpha, self.beta

    def derl(self, X=None, age=1, alive=1, arpu=1.0, discount_rate=0.005):
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
        params = self.predict(X)

        alpha = params[:, 0]
        beta = params[:, 1]

        # To make it so that the formula resembles that of the paper we define
        # the parameter n as below.
        n = age

        # The equation is two long, so we break in two parts.
        f1 = (beta + n - 1) / (alpha + beta + n - 1)
        f2 = hyp2f1(1., beta + n, alpha + beta + n, 1. / (1. + discount_rate))

        return arpu * f1 * f2 * alive

    def churn_p_of_t(self, X=None, age=1, alive=1, n_periods=12):
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

        # Load alpha and beta sampled from the posterior. These fully
        # determine the beta distribution governing the customer level
        # churn rates
        params = self.predict(X)

        alpha = params[:, 0]
        beta = params[:, 1]

        # --- Initialize Output ---
        # Initialize the output as a matrix of zeros. The number of rows is
        # given by the total number of samples, while the number of columns
        # is the number of months passed as a parameter.
        p_churn_matrix = numpy.zeros((self.n_samples, n_periods))

        # --- Fill output recursively (see eq.7 in [1])---

        # Start with month one (churn rate of month zero was set to 0 by
        # definition).
        p_churn_matrix[:, 1] = alpha / (alpha + beta)

        # Calculate remaining months recursively using the formulas
        # provided in the original paper.
        for i in range(2, n_periods):

            month = i
            update = (beta + month - 2) / (alpha + beta + month - 1)

            # None that i + 1 is simply the previous value, since val
            # starts with the third entry in the array, but I starts
            # counting form zero!
            p_churn_matrix[:, i] += update * p_churn_matrix[:, i - 1]

        return p_churn_matrix

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
