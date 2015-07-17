from __future__ import print_function
from scipy.optimize import minimize
from math import log10, exp
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
                 gamma_beta=1.0,
                 verbose=False):
        """
        This object is initialized with training time hyper-parameters and a
        verbose option.

        :param gamma_alpha: float
            A non-negative float specifying the strength of the regularization
            applied to w_alpha (alpha's weights).

        :param gamma_beta: float
            A non-negative float specifying the strength of the regularization
            applied to w_beta (beta's weights).

        :param verbose: bool
            Whether of not status updates should be printed
        """

        # --- Parameters ---
        # alpha and beta are the parameters learned by this model. When the
        # time is right they will be arrays with length a function of number of
        # predictors and whether or not a bias is being used. For now we
        # create a place holder array of a single zero.#
        self.alpha = numpy.zeros(1)
        self.beta = numpy.zeros(1)

        # --- Regularization ---
        # In this model regularization helps by both limiting the model's
        # complexity as well as greatly improving numerical stability during
        # the optimization process.
        # Moreover different regularization parameters for alpha and beta
        # can be helpful, specially in extreme cases when the distribution
        # is near the extremes (0 or 1).

        # Clearly both gammas must be non-negative, so we make sure to check
        # for it here.
        if gamma_alpha < 0:
            raise ValueError("The regularization constant gamma_alpha must "
                             "be a non-negative real number. A negative "
                             "value of {} was passed.".format(gamma_alpha))

        if gamma_beta < 0:
            raise ValueError("The regularization constant gamma_beta must "
                             "be a non-negative real number. A negative "
                             "value of {} was passed.".format(gamma_beta))

        self.gammaa = gamma_alpha
        self.gammab = gamma_beta

        # Boolean variable controling whether or not status updates should be
        # printed during training and other stages.
        self.verbose = verbose

        # A variable to store the size of the dataset, useful in certain
        # situations where no predictors are being used.
        self.n_samples = 0

    @staticmethod
    def _recursive_retention_stats(alpha, beta, num_periods):
        """
        A function to calculate the expected probabilities recursively.
        Using equation 7 from [1] and the alpha and beta coefficients
        obtained by training this model, it computes P(T = t)  as well
        as S(T = t) recursively, returning only the relevant values
        for computing the individual contribution to the likelihood.

        :param alpha: float
            A value for the alpha parameter.

        :param beta: float
            A value for the beta parameter.

        :param num_periods: Int
            The number of periods for which the probability of churning
            should be computed.

        :return: (float, float)
            A tuple with both the probability of dieing as well as
            surviving the current period.
        """
        # Extreme values of alpha and beta can cause severe numerical stability
        # problems! We avoid some of if by clipping the values of both alpha and
        # beta parameters such that they lie between 1e-5 and 1e5.
        alpha = max(min(alpha, 1e5), 1e-5)
        beta = max(min(beta, 1e5), 1e-5)

        # --- Initialize Recursion Values
        # We hold off initializing p_old since it is not necessary until we
        # enter the loop. s_old is initialized to 1, as it should.
        s_old = 1.

        # Accoring to equation 7 in [1] the next values of p and s are given by
        p_new = alpha / (alpha + beta)
        s_new = 1. - p_new

        # For subsequent periods we calculate the new values of both p and s
        # recursively. Updating old and new values accordingly.
        #
        # ** Note that the loop starts with a value of two and extends to
        # num_periods + 1, the reason behind this is that num_periods will
        # usually represent the age of a subject. In the context of a
        # subscription based business, the age of a subject often translates to
        # how many payments have been made so far, and thus must be a positive
        # integer.
        # A value of one means the subject is in its first period and
        # the entire population is still alive (by definition), hence the
        # values initialized prior to the loop. **
        for t in range(2, num_periods + 1):

            # Update old values with current new values
            p_old = 1. * p_new
            s_old = 1. * s_new

            # Update p_new with the latest p value
            p_new = (beta + t - 2.) / (alpha + beta + t - 1.) * p_old

            # Use the newly calculated p_new to update s_new
            s_new = s_old - p_new

        # Note that p_new is the likelihood of not making to the next period
        # while s_old is the likelihood of surviving the current period. Which
        # is used in the likelihood depends on whether or not the subject is
        # alive or dead.
        return p_new, s_old

    @staticmethod
    def _compute_alpha_beta(X, w_a, w_b):
        """
        This method computes the float values of alpha and beta given a matrix
        of predictors X and an array of weighs wa and wb. It does so by taking
        the dot product of w_a (w_b) with the matrix X and exponentiating it
        the resulting array.

        :param X: ndarray of shape (n_samples, n_features)
            The feature matrix.

        :param w_a: ndarray of shape (n_features, )
            Array of weights for the alpha parameter

        :param w_b: ndarray of shape (n_features, )
            Array of weights for the alpha parameter

        :return: (ndarray, ndarray) both with shapes (n_samples, )
            The alpha and beta values calculated for each row of the feature
            matrix X.
        """

        # Take that dot product!
        waT_dot_X = (w_a * X).sum(axis=1)
        wbT_dot_X = (w_b * X).sum(axis=1)

        # Return the element-wise exponential
        return numpy.exp(waT_dot_X), numpy.exp(wbT_dot_X)

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
        # In this model regularization helps by both limiting the model's
        # complexity as well as greatly improving numerical stability during
        # the optimization process.
        # However, it is undesirable to regularize the bias weights, since
        # this can stop the model from learning anything. *** Note that
        # this is different than the case of, say, a linear model, where the
        # trivial model (with zero weights) approximates the mean of the
        # target values. Here, the absence of weights (including bias) does
        # NOT lead to a trivial model, but one with a unreasonable
        # preference for alpha = exp(0) and beta = exp(0). ***
        # Moreover different regularization parameters for alpha and beta
        # can be helpful, specially in extreme cases when the distribution
        # is near the extremes (0 or 1)
        l2_reg = self.gammaa * sum(wa[1:]**2) + self.gammab * sum(wb[1:]**2)

        # update log-likelihood with regularization val.
        log_like -= l2_reg

        # get real alpha and beta
        alpha, beta = self._compute_alpha_beta(X, wa, wb)

        # loop over data doing stuff
        for y, z, a, b in zip(age, alive, alpha, beta):

            # add contribution of current customer to likelihood
            log_like += numpy.log(self._recursive_retention_stats(a, b, y)[z])

        # Negative log_like since we will use scipy's minimize object.
        return -log_like

    def fit(self, X, age, alive, restarts=1):
        """
        Method responsible for the learning step it takes all the relevant data
        as argument as well as the number of restarts with random seeds to
        perform. While restarting with other seeds sounds like a good idea the
        model has proven to be fairly stable and this may be removed in the
        future.

        *** This model can work without any features (only bias), think about
        the best way to integrate this into the code! ***

        :param X: ndarray of shape (n_samples, n_features)
            The feature matrix

        :param age: ndarray of shape (n_samples, )
            An array with the age of each individual.

        :param alive: ndarray of shape (n_samples, )
            An array with
        :param restarts:
        :return:
        """

        # Make sure ages are all non negative...
        min_age = min(age)
        if min(age) < min_age:
            raise ValueError("All values of age must be equal or greater to "
                             "one. The minimum value of "
                             "{} was found.".format(min_age))

        # Make sure alive is either zero or one!
        alive_vals = set(alive)
        if alive_vals != {0, 1}:
            raise ValueError('Values for alive must be either zero or one. A '
                             'value of '
                             '{} was found.'.format(list(alive_vals - {0, 1})))

        # Free space when printing based on the total amount of restarts.
        print_space = max(int(log10(restarts)) + 1, 5)

        # store the number of samples we are dealing with
        self.n_samples = age.shape[0]

        # Now that we have X we can calculate the number of params we need
        n_params = X.shape[1]

        # --- Optimization Starting Points
        # Generate random starting points for optimization step.
        initial_guesses = 0.1 * numpy.random.randn(restarts, 2 * n_params) - 0.01

        # Initialize optimal value to None
        # I choose not to set it to, say, zero, or any other number, since I am
        # not sure that the log-likelihood is bounded in anyway. So is better to
        # initialize with None and use the first optimal value to get the ball
        # rolling.
        optimal = None
        opt_params = numpy.zeros((2 * n_params))

        # clock
        start = datetime.now()

        # Print a nice looking  header for the optimization process
        if self.verbose:
            print('Starting Optimization with parameters:')
            print('{:>15}: {}'.format('Samples', self.n_samples))
            print('{:>15}: {}'.format('gamma (alpha)', self.gammaa))
            print('{:>15}: {}'.format('gamma (beta)', self.gammaa))
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
            # --- Variables
            #   step: Integer - current step number
            #  guess: Array - array with starting points for minimization

            # --- Optimization
            # Unbounded optimization (minimization) of negative log-likelihood
            new_opt = minimize(lambda p: self._logp(X=X,
                                                    age=age,
                                                    alive=alive,
                                                    wa=p[:n_params],
                                                    wb=p[n_params:]),
                               guess,
                               bounds=[(None, None)] * n_params * 2
                               )

            # For the first run only optimal is None, that being the case, we
            # set the current values to both optimal (function value) as well
            # as opt_params - parameters that minimize the function.
            if optimal is None:
                optimal = new_opt.fun
                opt_params = new_opt.x

            # Have we found a better value yet? If we have, update optimal
            # with new function minimum and opt_params with corresponding
            # minimizing parameters.
            if new_opt.fun < optimal:
                optimal = new_opt.fun
                opt_params = new_opt.x

            # Print current status if verbose is True.
            if self.verbose:
                print_string = "{0: {3}} | {1:^10.10} | {2:13.8} |"
                print(print_string.format(step + 1,
                                          datetime.now() - start,
                                          optimal,
                                          print_space))

        # --- Parameter Values
        # Optimization is complete, time to save best parameters.
        # Note that we breakdown the parameters passed to and returned from
        # the scipy.optimize.minimize object in two. The first half correspond
        # to the alpha parameter, while the second half is beta.
        self.alpha = opt_params[:n_params]
        self.beta = opt_params[n_params:]

        # --- Regularization Penalty
        # Compute the regularization penalty applied to the parameter vectors.
        # Remember that there is no penalty for bias!
        reg_penalty = self.gammaa * sum(self.alpha[1:]**2) + \
            self.gammab * sum(self.beta[1:]**2)

        # Print some final remarks before we say goodbye.
        if self.verbose:
            print()
            print('Optimization completed:')
            print('{:>15}: {}'.format('wa', self.alpha))
            print('{:>15}: {}'.format('wb', self.beta))
            print('{:>15}: {}'.format('LogLikelihood', optimal))
            print('{:>15}: {}'.format('Reg. Penalty', reg_penalty))
            print()

    def derl(self,
             X,
             age=1,
             alive=1,
             arpu=1.0,
             discount_rate=0.005):
        """
        Discounted Expected Residual Lifetime, as derived in [2].
        See equation (6).

        :param X:
        :param age:
        :param alive:
        :param arpu:
        :param discount_rate:

        :return: DERL
        """

        alpha, beta = self._compute_alpha_beta(X, self.alpha, self.beta)

        # To make it so that the formula resembles that of the paper we define
        # the parameter n as below.
        n = age

        # The equation is two long, so we break in two parts.
        f1 = (beta + n - 1) / (alpha + beta + n - 1)
        f2 = hyp2f1(1., beta + n, alpha + beta + n, 1. / (1. + discount_rate))

        return arpu * f1 * f2 * alive

    def churn_p_of_t(self,
                     X,
                     age=1,
                     n_periods=12):
        """
        churn_p_of_t computes the churn as a function of time curve. Using
        equation 7 from [1] and the alpha and beta coefficients obtained by
        training this model, it computes P(T = t) recursively, returning either
        the expected value or an array of values.

        :param X:
        :param age:
        :param n_periods:
        :return:
        """

        # Spot checks making sure the values passed make sense!
        if n_periods < 0 or not isinstance(n_periods, int):
            raise ValueError("The number of periods must be a non-zero "
                             "integer")

        # Load alpha and beta sampled from the posterior. These fully
        # determine the beta distribution governing the customer level
        # churn rates
        alpha, beta = self._compute_alpha_beta(X, self.alpha, self.beta)

        # set the number of samples
        n_samples = X.shape[0]

        try:
            len(age) == X.shape[0]
        except TypeError:
            age = age * numpy.ones(n_samples, dtype=int)

        # age cannot be negative!
        if min(age) < 0:
            raise ValueError("All ages must be non-negative.")

        # --- Initialize Output ---
        # Initialize the output as a matrix of zeros. The number of rows is
        # given by the total number of samples, while the number of columns
        # is the number of months passed as a parameter.
        p_churn_matrix = numpy.zeros((n_samples, max(age) + n_periods))
        outputs = numpy.zeros((n_samples, max(age) + n_periods),
                              dtype=bool)

        # sort this whole age thing out!
        outputs[:, 0][age < 1] = True
        outputs[:, 1][age < 2] = True

        # --- Fill output recursively (see eq.7 in [1]) ---

        # Start with month one (churn rate of month zero was set to 0 by
        # definition).
        p_churn_matrix[:, 1] = alpha / (alpha + beta)

        # Calculate remaining months recursively using the formulas
        # provided in the original paper.
        for period in range(2, max(age) + n_periods):

            month = period
            update = (beta + month - 2) / (alpha + beta + month - 1)

            # None that i + 1 is simply the previous value, since val
            # starts with the third entry in the array, but I starts
            # counting form zero!
            p_churn_matrix[:, period] += update * p_churn_matrix[:, period - 1]

            # correct rows
            rows = (period >= age) & (period < (age + n_periods))
            outputs[:, period][rows] += 1

        # Return only the appropriate values and reshape the matrix.
        return p_churn_matrix[outputs].reshape((n_samples, n_periods))

    def survival_function(self, X, age=1,  n_periods=12):
        """
        survival_function computes the survival curve obtained from the model's
        parameters and assumptions. Using equation 7 from [1] and the alpha and
        beta coefficients obtained by training this model, it computes S(T = t)
        recursively, returning either the expected value or an array of values.
        To do so it must first invoke the self.churn_p_of_t method to calculate
        the monthly churn rates for the given time window, and then use it to
        compute the survival curve recursively.

        :param X:
        :param age:
        :param n_periods:

        :return:
        """

        # Spot checks making sure the values passed make sense!
        if n_periods < 0 or not isinstance(n_periods, int):
            raise ValueError("The number of periods must be a non-zero "
                             "integer")

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

        # set the number of samples
        n_samples = X.shape[0]

        # get number of periods
        try:
            num_periods = int(max(age) + n_periods)

            # is age a list like object?
            len(age) == X.shape[0]
        except TypeError:
            age = age * numpy.ones(n_samples, dtype=int)
            num_periods = int(max(age) + n_periods)

        # age cannot be negative!
        if min(age) < 0:
            raise ValueError("All ages must be non-negative.")

        # Age of zero means we want survival func of current month as the
        # starting point! explain more.
        p_of_t = self.churn_p_of_t(X=X,
                                   age=0,
                                   n_periods=num_periods)

        s = numpy.zeros(p_of_t.shape)
        s[:, 0] = 1.

        # output bool mask
        outputs = numpy.zeros(p_of_t.shape,
                              dtype=bool)

        # set initial values
        outputs[:, 0][age < 1] = True
        outputs[:, 1][age < 2] = True

        for col in range(1, s.shape[1]):
            s[:, col] = s[:, col - 1] - p_of_t[:, col]

            # correct rows
            rows = (col >= age) & (col < (age + n_periods))
            outputs[:, col][rows] += 1

        # pick correct entries
        s = s[outputs].reshape((n_samples, n_periods))

        # return the scaled values of s
        return s/s[:, [0]]
