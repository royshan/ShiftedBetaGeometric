from scipy.optimize import minimize
import numpy


class ShiftedBeta:

    def __init__(self, data):

        self.data = data

        self.opt = None
        self.alpha = 0.1
        self.beta = 0.1

    @staticmethod
    def _P_T_is_t(alpha, beta, num_periods):
        """
        A function to calculate the expected probabilities recursively.
        Using equation 7 from [1] and the alpha and beta coefficients
        obtained by training this model, it computes P(T = t) recursively,
        returning a list with all values.

        :param alpha: PyMC distribution
            The distribution for the alpha parameter.

        :param beta: PyMC distribution
            The distribution for the beta parameter.

        :param num_periods: Int
            The number of periods for which the probability of churning
            should be computed.

        :return: List
            A list with probability of churning for all periods from month
            zero to num_periods.
        """

        # Initialize list with t = 0 and t = 1 values
        p = [None, alpha / (alpha + beta)]

        for t in range(2, num_periods):
            pt = (beta + t - 2) / (alpha + beta + t - 1) * p[t-1]
            p.append(pt)

        return p

    @staticmethod
    def _survival_function(P_T_is_t):
        """
        Survival function recursive calculation. Using equation 7 from [1]
        and the alpha and beta coefficients obtained by training this
        model, it computes S(T = t) recursively, returning a list of all
        computed values.. To do so it must first invoke the function
        P_T_is_t calculate the monthly churn rates for the given time
        window, and then use it to compute the survival curve recursively.

        :param P_T_is_t: Deterministic Variable
            The PyMC deterministic variable defined above.

        :param num_periods: Int
            The number of periods for which the probability of churning
            should be computed.

        :return: List
            A list with values of the survival functions for all periods
            from month zero to num_periods.
        """

        # Initial values
        s = [None, 1 - P_T_is_t[1]]

        for t in range(2, len(P_T_is_t)):
            s.append(s[t-1] - P_T_is_t[t])

        return s

    def _logp(self, alpha, beta):
        """
        The LogLikelihood function. Given the data and relevant
        variables this function computed the loglikelihood.

        :param value: List
            The formatted data set --- a list of lists with augmented
            cohort information.

        :param P_T_is_t: Deterministic Variable
            The PyMC deterministic variable defined above.

        :param survival_function: Deterministic Variable
            The PyMC deterministic variable defined above.

        :param sizes: List
            A List with the true size of the cohort data. This is used to
            keep the model from using the augmented data in the formatted
            dataset (necessary only to please PyMC and simplify life).

        :return: Float
            The LogLikelihood of the model.
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

        # A loop through each element in the data list. Remember that
        # each element correspond to a particular cohort data. The loop
        # simply carries out the calculation in B3, appendix B, [1].
        for i, val in enumerate(self.data):

            # The number of customer that are still active and the
            # number of customers lost at each month for which cohort
            # data is available.
            active, lost = val

            # Since the original dataset was augmented earlier in this
            # method, we must specify the point at which the
            # calculations performed here should stop. In other words,
            # length indicates the point at which actual data is
            # available.
            length = len(active) - 1

            # stuff...#
            pt = self._P_T_is_t(alpha, beta, length + 1)
            sf = self._survival_function(pt)

            # Likelihood of observing such data given the model.
            # Refer to equation B3 for context.
            # *** Note that the data is used only up to index length,
            # hence avoiding the inclusion of augmented data points.
            # ***
            died = numpy.log(pt[1:length + 1]) * lost[1:length + 1]

            # Likelihood of having this many people left after
            # some time
            still_active = numpy.log(sf[length]) * active[length]

            # Update the log_like value.
            log_like += sum(died) + still_active

        return -log_like

    def fit(self):
        self.opt = minimize(lambda p: self._logp(p[0], p[1]),
                            [self.alpha, self.beta],
                            bounds=((0, None), (0, None)))

        self.alpha = self.opt.x[0]
        self.beta = self.opt.x[1]