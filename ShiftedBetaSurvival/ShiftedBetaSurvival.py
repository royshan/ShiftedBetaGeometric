from DataHandler import DataHandler
from ShiftedBeta import ShiftedBeta
import numpy
import pandas
from math import exp
from scipy.special import hyp2f1


class ShiftedBetaSurvival(object):

    def __init__(self, verbose=False):

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

        # verbose controler
        self.verbose = verbose

    def fit(self, df, cohort, age, category=None, restarts=50):

        # Set a bunch of instance parameters
        self.df = df
        self.cohort = cohort
        self.age = age
        self.category = category

        # Create datahandler object
        self.dh = DataHandler(data=self.df,
                              cohort=self.cohort,
                              age=self.age,
                              category=self.category)

        self.data = self.dh.paired_data()

        # create shifted beta object
        self.sb = ShiftedBeta(self.data, verbose=self.verbose)

        # fit to data!
        self.sb.fit(restarts=restarts)
        self.sb_params = self.sb.get_params()

        # Trained successful means training is done!
        self.trained = True

    def summary(self):
        """

        :return:
        """

        # Model ran?
        if not self.trained:
            raise RuntimeError('Train the model first!')

        # Some info
        out = pandas.DataFrame(columns=['Category',
                                        'Value',
                                        'Coefficient Alpha',
                                        'Coefficient Beta',
                                        'Alpha',
                                        'Beta',
                                        'Avg Churn'])

        row = 0
        # category loop
        for category, val_list in self.sb.get_params()['categories'].items():

            # value loop
            for value in val_list:

                alpha = self.sb.get_coeffs()[category][value]['alpha']
                beta = self.sb.get_coeffs()[category][value]['beta']

                alpha_coeff = self.sb_params['coeffs']['alpha'][self.sb_params['imap'][category][value]][-1]
                beta_coeff = self.sb_params['coeffs']['beta'][self.sb_params['imap'][category][value]][-1]

                # Populate dataframe
                out.loc[row] = [category,
                                value,
                                alpha_coeff,
                                beta_coeff,
                                alpha,
                                beta,
                                alpha / (alpha + beta)]

                # Next row, please!
                row += 1
        return out

    def get_coeffs(self):

        if not self.trained:
            raise RuntimeError('Train the model first!')

        return self.sb.get_coeffs()

    def _coefficients_combination(self):

        categories = sorted(self.sb_params['categories'])
        print categories

        for i, category_1 in enumerate(categories):
            for j, category_2 in enumerate(categories[i + 1:]):

                #for k, value_1 in

                combo = category_1 + "_" + category_2

                print combo

        return 0

    def _predict_coefficients(self, row):
        """

        :param row:
        :return:
        """

        # bool map to pick up correct coefficients
        bool_map = numpy.zeros(self.sb_params['n_categories'], dtype=bool)

        for category in self.category:
            # turn appropriate entries to True
            bool_map[self.sb_params['imap'][category][row[category]]] = True

        # log of sum of coeffs
        log_alpha = self.sb_params['coeffs']['alpha'][bool_map].sum()
        log_beta = self.sb_params['coeffs']['beta'][bool_map].sum()

        # return values
        return dict(alpha=exp(log_alpha), beta=exp(log_beta))

    def _predict_row(self, row, **kwargs):
        """

        :param row:
        :param kwargs:
        :return:
        """
        params = self._predict_coefficients(row)
        return self.derl(alpha=params['alpha'], beta=params['beta'], **kwargs)

    def predict_ltv(self, df, **kwargs):
        """

        :param df:
        :param kwargs:
        :return:
        """
        return df.apply(lambda row: self._predict_row(row, **kwargs), axis=1).values

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

        # The usual empty dict to hold a dict of dicts of results
        ltv_by_cate = {}

        # category loop
        for category, val_dict in self.sb.get_coeffs().items():

            # add category dict
            ltv_by_cate[category] = {}

            for value, param in val_dict.items():

                ltv_by_cate[category][value] = \
                    self.derl(alpha=param['alpha'],
                              beta=param['beta'],
                              arpu=arpu,
                              discount_rate=discount_rate,
                              renewals=renewals)

        return ltv_by_cate

    def capped_ltv(self):
        if not self.trained:
            raise RuntimeError('Train the model first!')
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
