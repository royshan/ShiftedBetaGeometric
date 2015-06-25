from DataHandler import DataHandler
from ShiftedBeta import ShiftedBeta
import numpy
import pandas
from scipy.special import hyp2f1


class ShiftedBetaSurvival(object):

    def __init__(self, verbose=0):

        # ShiftedBeta()
        self.sb = None
        self.sb_params = None
        # DataHandler()
        self.dh = None

        self.df = None
        self.data = None
        self.cohort = None
        self.age = None
        self.category = None

        # trained?
        self.trained = False

        # params
        self.alpha = None
        self.beta = None

    def fit(self, df, cohort, age, category=None, restarts=50):

        self.df = df
        self.cohort = cohort
        self.age = age
        self.category = category

        self.dh = DataHandler(data=self.df,
                              cohort=self.cohort,
                              age=self.age,
                              category=self.category)

        self.data = self.dh.paired_data()

        self.sb = ShiftedBeta(self.data)

        self.sb.fit(restarts=restarts)
        self.sb_params = self.sb.get_params()

        # Trained successful means training is done!
        self.trained = True

    def summary(self, print_res=True):
        """

        :param print_res:
        :return:
        """

        # !!!!!!!
        # Get this method to adapt its size to always fit the data!
        # !!!!!!!!!!!!!!
        if not self.trained:
            raise RuntimeError('Train the model first!')

        pdict = self.sb.get_coeffs()
        cate_list = sorted(pdict.keys())

        out = pandas.DataFrame(columns=['Category',
                                        'Alpha',
                                        'Beta',
                                        'Avg Churn'])

        for row, cate in enumerate(cate_list):

            out.loc[row] = [cate,
                        pdict[cate]['alpha'],
                        pdict[cate]['beta'],
                        pdict[cate]['alpha'] /
                        (pdict[cate]['alpha'] + pdict[cate]['beta'])]

        if print_res:
            print out
        else:
            return out

    def get_coeffs(self):

        if not self.trained:
            raise RuntimeError('Train the model first!')

        return self.sb.get_coeffs()

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

        for cate, val in self.sb.get_coeffs().items():

            # Load alpha and beta sampled from the posterior. These fully
            # determine the beta distribution governing the customer level
            # churn rates
            alpha = val['alpha']
            beta = val['beta']

            # --- Initialize Output ---
            # Initialize the output as a matrix of zeros. The number of rows is
            # given by the total number of samples, while the number of columns
            # is the number of months passed as a parameter.
            churn_by_cate[cate] = numpy.zeros(n_periods)

            # --- Fill output recursively (see eq.7 in [1])---

            # Start with month one (churn rate of month zero was set to 0 by
            # definition).
            churn_by_cate[cate][1] = alpha / (alpha + beta)

            # Calculate remaining months recursively using the formulas
            # provided in the original paper.
            for i in range(2, n_periods):

                month = i
                update = (beta + month - 2) / (alpha + beta + month - 1)

                # None that i + 1 is simply the previous value, since val
                # starts with the third entry in the array, but I starts
                # counting form zero!
                churn_by_cate[cate][i] += update * churn_by_cate[cate][i - 1]

        return pandas.DataFrame(data=churn_by_cate)

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

        for cate in self.sb_params['categories']:

            # --- Initialize output ---
            # The output is initialized as a zero matrix with the same shape
            # as the churn rates matrix
            surv_by_cate[cate] = numpy.zeros(p_of_t.shape[0])

            # The initial value is one by definition (in this model death at
            # t=0 is no considered).
            surv_by_cate[cate][0] = 1

            # The value of month is simply given by the naive formula
            #       1 - churn(t=1)
            surv_by_cate[cate][1] = 1 - p_of_t[cate].values[1]

            # The remaining values are calculated recursively using eq. 7 [1].
            for i, val in enumerate(p_of_t[cate].values[2:]):

                # Something here...
                surv_by_cate[cate][i + 2] = surv_by_cate[cate][i + 1] - val

        # To data-frame and some re-formatting
        out = pandas.DataFrame(data=surv_by_cate)
        out = out.iloc[renewals:]
        out.index = range(out.shape[0])

        return out

    def ltv(self, arpu=1.0, discount_rate=0.005, renewals=0):
        """
        This method calculates the full residual LTV given the model's
        parameters alpha, beta in addition to arpu, discount_rate, number of
        renewals.

        It uses the DERL equation derived in [2] to obtain the residual tenure
        of a customer.

        This function may return either the full distribution for the residual
        LTV given alpha's and beta's distribution, or, instead, the expected
        value.

        :param arpu: Float
            A flat value for the average revenue per user to be used by the
            DERL function.

        :param discount_rate: Float
            A fixed discounted rate.

        :param renewals: Int
            Number of times the customer has renewed his or her subscription.

        :return: Float / ndarray
            Either a float with the expected value for the residual LTV or a
            ndarray with the distribution given alpha and beta.
        """
        # Spot checks making sure the values passed make sense!
        if arpu < 0:
            raise ValueError("ARPU must be a non-negative number.")

        if discount_rate <= 0:
            raise ValueError("The discount rate must be a positive number.")

        if renewals < 0 or not isinstance(renewals, int):
            raise ValueError("The number of renewals must be a non-zero "
                             "integer")

        if not self.trained:
            raise RuntimeError('Train the model first!')

        ltv_by_cate = {}

        for cate, val in self.sb.get_coeffs().items():

            ltv_by_cate[cate] = self.derl(alpha=val['alpha'],
                                          beta=val['beta'],
                                          arpu=arpu,
                                          discount_rate=discount_rate,
                                          renewals=renewals)

        return ltv_by_cate

    def capped_ltv(self):
        print('Not implemented yet, use ltv instead.')

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
