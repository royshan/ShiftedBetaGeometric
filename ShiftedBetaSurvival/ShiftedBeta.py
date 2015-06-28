from __future__ import print_function
from scipy.optimize import minimize
from math import log10
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

    def __init__(self, data, gamma=1.0, verbose=False):
        """

        :param data:
        :param verbose:
        :return:
        """

        self.data = data

        self.categories = {}
        self.n_cats = 0
        # What is this? Why I need this?
        for category in sorted(data.keys()):
            # enter cat
            self.categories[category] = []

            for value in sorted(data[category].keys()):
                self.categories[category].append(value)
                self.n_cats += 1

        # params constructor, explain!
        self.imap = {}
        # construct imap by looping over all category-value combination and
        # setting a unique boolean array to it. This array will dictate the
        # combination of weights used in the linear model for this pair.
        self.imap_constructor()

        # ps
        self.alpha = {}
        self.beta = {}

        self.alpha_coeffs = None
        self.beta_coeffs = None

        # regularizer
        if gamma < 0:
            raise ValueError("The regularization constant gamma must be a "
                             "non-negative real number. A negative value of"
                             " {} was passed.".format(gamma))
        self.gamma = gamma

        # ops obj
        self.opt = None

        # verbose param
        self.verbose = verbose

    def imap_constructor(self):
        """
        indicator_map constructs a boolean vector indicating which parameters
        to use for a given predictor.

        alpha and beta paramaters are assumed to be linear combination like:

            alpha = alpha0 + alpha1 * predictor1 + alpha2 * predictor2 + ...

        and similarly for beta. However, as it stands, predictors are
        one-hot encoded categorical variables, so at any given time at most
        only two alpha_i are used, the intercept and coefficient of the
        current predictor. The indicator_map methods takes care of keeping
        track of that.
        """

        # Easier to use an index that is added one
        index = 0

        # For each category in the data turn on a different combination of a
        # boolean array.
        for category, values_list in self.categories.items():

            # For each category in the data we turn on a different
            # combination of a boolean array.
            self.imap[category] = {}

            for value in values_list:

                # Initial a boolean array as false, with length equal to the
                # number of available categories.
                bool_ind = numpy.zeros(self.n_cats, dtype=bool)

                # The intercept (index = 0) is always on.
                bool_ind[0] = True

                # For any category-value combination but the first, both the
                # intercept as well as an extra entry are set to True.
                bool_ind[index] = True

                # Change the instance variable imap in place by adding the
                # appropriate key: bool array pair.
                self.imap[category][value] = bool_ind

                # add to index
                index += 1

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

        # Initialize list with t = 0 and t = 1 values
        p = [None, alpha / (alpha + beta)]
        s = [None, 1. - p[1]]

        for t in range(2, num_periods):
            # Compute latest p value and appen
            pt = (beta + t - 2.) / (alpha + beta + t - 1.) * p[t-1]
            p.append(pt)

            # use the most recent appended p value to keep building s
            s.append(s[t-1] - p[t])

        # finish this...
        return p, s

    def _logp(self, alpha, beta):
        """
        The LogLikelihood function. Given the data and relevant
        variables this function computed the loglikelihood.

        :param alpha: array
            aaa

        :param alpha: array
            bbb

        :return: Float
            Minus the LogLikelihood of the model.
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

        # loop across categories
        for category, val_dicts in self.data.items():

            # Loop across values of these categories
            for value, data in val_dicts.items():

                bool_ind = self.imap[category][value]

                alpha_comb = numpy.exp(alpha[bool_ind].sum())
                beta_comb = numpy.exp(beta[bool_ind].sum())

                # A loop through each element in the data list. Remember that
                # each element correspond to a particular cohort data. The loop
                # simply carries out the calculation in B3, appendix B, [1].
                for i, val in enumerate(data):

                    # The number of customer that are still active and the
                    # number of customers lost at each month for which cohort
                    # data is available.
                    active, lost = val

                    # Since the original dataset was augmented earlier in this
                    # method, we must specify the point at which the
                    # calculations performed here should stop. In other words,
                    # length indicates the point at which actual data is
                    # available.
                    length = len(active)

                    # stuff...#
                    pt, sf = self._recursive_retention_stats(alpha=alpha_comb,
                                                             beta=beta_comb,
                                                             num_periods=length)

                    # Likelihood of observing such data given the model.
                    # Refer to equation B3 for context.
                    # *** Note that the data is used only up to index length,
                    # hence avoiding the inclusion of augmented data points.
                    # ***
                    died = numpy.log(pt[1:length]) * lost[1:length]

                    # Likelihood of having this many people left after
                    # some time
                    still_active = numpy.log(sf[length - 1]) * active[length - 1]

                    # Update the log_like value.
                    log_like += sum(died) + still_active

        # L2 regularizer
        # something about l2 regularization
        # Explain why the intercept (zero-th index portion of alpha and beta)
        # are not subject to regularization. Also, think whether this is the
        # best way of handling this, or whether adding a dedicated intercept
        # is a better choice.
        l2_reg = self.gamma * (sum(alpha[1:]**2) + sum(beta[1:]**2))

        # Negative log_like since we will use scipy's minimize object.
        return -(log_like - l2_reg)

    def fit(self, restarts=50):
        """

        :param restarts:
        :return:
        """
        # Free space when printing
        print_space = int(log10(restarts)) + 1

        # guesses of initial parameters
        initial_guesses = 4 * numpy.random.random((restarts, 2 * self.n_cats)) - 3

        # Initialize optimal value to None
        # I choose not to set it a, say, zero, or any other number, since I am
        # not sure that the log-likelihood is bounded in anyway. So is better to
        # initialize with None and use the first optimal value start the ball
        # rolling.
        optimal = None

        # Run likelihood optimization for several steps...
        # noinspection PyTypeChecker
        for step, guess in enumerate(initial_guesses):

            # --- Optimization
            # something...
            new_opt = minimize(lambda p: self._logp(p[:self.n_cats],
                                                    p[self.n_cats:]),
                               guess,
                               bounds=[(None, None)] * 2 * self.n_cats
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

        # --- Update values of alpha and beta related coefficients ---

        # The full, raw coefficient arrays
        self.alpha_coeffs = self.opt[:self.n_cats]
        self.beta_coeffs = self.opt[self.n_cats:]

        # Categories and their corresponding values
        for category, val_list in self.categories.items():

            # Initialize with empty dict
            self.alpha[category] = {}
            self.beta[category] = {}

            # Values list
            for value in val_list:

                # Is boolean ideal?
                bool_ind = self.imap[category][value]

                self.alpha[category][value] = numpy.exp(self.opt[:self.n_cats][bool_ind].sum())
                self.beta[category][value] = numpy.exp(self.opt[self.n_cats:][bool_ind].sum())

    def get_coeffs(self):
        """

        :return:
        """

        # gets alpha and beta...
        coeffs = {}

        # Categories and their corresponding values
        for category, val_list in self.categories.items():

            # Initialize with empty dict
            coeffs[category] = {}

            # Values list
            for value in val_list:
                coeffs[category][value] = dict(alpha=self.alpha[category][value],
                                               beta=self.beta[category][value])

        return coeffs

    def get_params(self):
        """

        :return:
        """

        # simple dict of stuff
        params = dict(n_categories=self.n_cats,
                      categories=self.categories,
                      imap=self.imap,
                      coeffs=dict(alpha=self.alpha_coeffs,
                                  beta=self.beta_coeffs)
                      )

        return params
