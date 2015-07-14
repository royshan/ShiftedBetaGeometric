from __future__ import print_function
from scipy.optimize import minimize
from math import log10
from datetime import datetime
import numpy


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
