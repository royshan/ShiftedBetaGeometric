from __future__ import print_function
from scipy.optimize import minimize
from math import log10, log
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

    def __init__(self, gamma_alpha=1.0, gamma_beta=1.0, add_bias=True, verbose=False):

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
        self.gammab = gamma_beta

        # bias
        self.bias = add_bias

        # ops obj
        self.opt = None

        # verbose param
        self.verbose = verbose

    @staticmethod
    def _recursive_retention_stats(alpha, beta, num_periods, alive):
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
        alpha = max(min(alpha, 25), 0.0001)
        beta = max(min(beta, 25), 0.0001)

        # Initialize list with t = 0 and t = 1 values
        p = [None, alpha / (alpha + beta)]
        s = [1., 1. - p[1]]

        for t in range(2, num_periods + 1):
            # Compute latest p value and append
            pt = (beta + t - 2.) / (alpha + beta + t - 1.) * p[t - 1]
            p.append(pt)

            # use the most recent appended p value to keep building s
            s.append(s[t - 1] - p[t])

        # finish this...
        if alive:
            # tricky mother fucker index! Pay a lot of attention and explain!!
            return s[-2]
        else:
            return p[-1]

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
            log_like += numpy.log(self._recursive_retention_stats(a, b, y, z))

        # Negative log_like since we will use scipy's minimize object.
        return -log_like

    def fit(self, age, alive, X=None, restarts=50):

        # Free space when printing
        print_space = int(log10(restarts)) + 1

        # Add bias
        if X is None:
            X = numpy.ones((age.shape[0], 1))
        elif self.bias:
            X = numpy.concatenate((numpy.ones((X.shape[0], 1)), X), axis=1)

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
                print("Maximization step "
                      "{0:{2}} of {1:{2}} completed".format(step + 1,
                                                            restarts,
                                                            print_space), end=" ")
                print("with LogLikelihood: {0}".format(optimal))

        self.alpha = self.opt[:n_params]
        self.beta = self.opt[n_params:]

    def get_params(self):
        return self.alpha, self.beta
