from scipy.optimize import minimize
import numpy


class ShiftedBeta:

    def __init__(self, data):

        self.data = data

        self.types = sorted(data.keys())
        self.n_types = len(data)

        # params constructor
        self.imap = {}
        self.indicator_map()

        # ps
        self.alpha = {}
        self.beta = {}

        # ops obj
        self.opt = None

    def indicator_map(self):
        for i, kind in enumerate(self.types):

            bool_ind = numpy.zeros(self.n_types, dtype=bool)
            bool_ind[0] = True
            bool_ind[i] = True

            self.imap[kind] = bool_ind

    @staticmethod
    def _p_t_is_t(alpha, beta, num_periods):
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
    def _survival_function(p_t_is_t):
        """
        Survival function recursive calculation. Using equation 7 from [1]
        and the alpha and beta coefficients obtained by training this
        model, it computes S(T = t) recursively, returning a list of all
        computed values.. To do so it must first invoke the function
        P_T_is_t calculate the monthly churn rates for the given time
        window, and then use it to compute the survival curve recursively.

        :param p_t_is_t: Deterministic Variable
            The PyMC deterministic variable defined above.

        :param num_periods: Int
            The number of periods for which the probability of churning
            should be computed.

        :return: List
            A list with values of the survival functions for all periods
            from month zero to num_periods.
        """

        # Initial values
        s = [None, 1 - p_t_is_t[1]]

        for t in range(2, len(p_t_is_t)):
            s.append(s[t-1] - p_t_is_t[t])

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

        for name, data in self.data.iteritems():

            bool_ind = self.imap[name]

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
                length = len(active) - 1

                # stuff...#
                pt = self._p_t_is_t(alpha_comb, beta_comb, length + 1)
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
        initial_guesses = 3*numpy.random.random((50, 2 * self.n_types)) - 2

        optimal = None

        for guess in initial_guesses:
            new_opt = minimize(lambda p: self._logp(p[:self.n_types], p[self.n_types:]),
                               guess,
                               bounds=[(None, None)] * 2 * self.n_types
                                       )

            if optimal is None:
                optimal = new_opt.fun
                self.opt = new_opt.x
            if new_opt > optimal:
                optimal = new_opt.fun
                self.opt = new_opt.x

        self.alpha = {}
        self.beta = {}

        for name in self.types:

            bool_ind = self.imap[name]

            self.alpha[name] = numpy.exp(self.opt[:self.n_types][bool_ind].sum())
            self.beta[name] = numpy.exp(self.opt[self.n_types:][bool_ind].sum())
