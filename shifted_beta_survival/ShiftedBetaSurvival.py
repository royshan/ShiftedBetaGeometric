from __future__ import print_function
from DataHandler import DataHandler
from ShiftedBeta import ShiftedBeta
import numpy as np
import pandas as pd


class ShiftedBetaSurvival(object):
    """
    what is this?
    """

    def __init__(self,
                 age,
                 alive,
                 features=None,
                 gamma=1.0,
                 gamma_beta=1.0,
                 bias=True,
                 normalize=True,
                 verbose=False):
        """

        :param age:
        :param alive:
        :param features:
        :param gamma:
        :param gamma_beta:
        :param bias:
        :param normalize:
        :param verbose:
        :return:
        """

        # Create objects!
        # DATA-HANDLER OBJECT
        # The DataHandler object may be created without the training data, so
        # we do it here.
        self.dh = DataHandler(age=age,
                              alive=alive,
                              features=features,
                              bias=bias,
                              normalize=normalize)

        # Shifted beta model object
        # create shifted beta object
        self.sb = ShiftedBeta(gamma_alpha=gamma,
                              gamma_beta=gamma_beta,
                              verbose=verbose)

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
        x, y, z = self.dh.fit_transform(df)

        # fit to data!
        self.sb.fit(X=x,
                    age=y,
                    alive=z,
                    restarts=restarts)

    def summary(self):
        """
        Simple method to get the learned weights and their corresponding
        categories

        :return: pandas DataFrame
            A DataFrame object with alpha and beta weights for each category
        """
        suma = pd.DataFrame(data={name: (a, b) for name, a, b in
                                  zip(self.dh.get_names(),
                                      self.sb.alpha,
                                      self.sb.beta)},
                            index=['w_alpha', 'w_beta']
                            ).T
        return suma

    def predict_params(self, df):
        """
        Predict alpha and beta for each sample

        :param df:
        :return:
        """
        x, y, z = self.dh.transform(df=df)
        alpha, beta = self.sb.compute_alpha_beta(x, self.sb.alpha, self.sb.beta)

        return pd.DataFrame(data=np.vstack([alpha, beta]),
                            index=['alpha', 'beta']).T

    def predict_churn(self, df, age=None, **kwargs):
        """
        Predict alpha and beta for each sample

        :param df:
        :return:
        """
        x, y, z = self.dh.transform(df=df)

        # If age field is present in prediction dataframe, we may choose to
        # use it to calculate future churn.
        if age is None:
            age = y
        if age is None:
            raise RuntimeError('The "age" field must either be present in '
                               'the dataframe or passed separately as an '
                               'argument.')

        out = pd.DataFrame(data=self.sb.churn_p_of_t(x, age=age, **kwargs))

        out.columns = ['period_{}'.format(col)
                       for col in range(1, out.shape[1] + 1)]

        return out

    def predict_survival(self, df, age=None, **kwargs):
        """
        Predict alpha and beta for each sample

        :param df:
        :return:
        """
        x, y, z = self.dh.transform(df=df)

        # If age field is present in prediction dataframe, we may choose to
        # use it to calculate future churn.
        if age is None:
            age = y
        if age is None:
            raise RuntimeError('The "age" field must either be present in '
                               'the dataframe or passed separately as an '
                               'argument.')

        out = pd.DataFrame(data=self.sb.survival_function(x,
                                                          age=age,
                                                          **kwargs))

        out.columns = ['period_{}'.format(col)
                       for col in range(1, out.shape[1] + 1)]

        return out

    def predict_ltv(self, df, age=None, alive=None, **kwargs):
        """

        :param df:
        :param kwargs:
        :return:
        """
        x, y, z = self.dh.transform(df=df)

        if age is None:
            age = y
        if age is None:
            raise RuntimeError('The "age" field must either be present in '
                               'the dataframe or passed separately as an '
                               'argument.')

        if alive is None:
            alive = z
        if alive is None:
            raise RuntimeError('The "alive" must either be present in the '
                               'dataframe or passed separately as an '
                               'argument.')

        ltvs = self.sb.derl(x, age=age, alive=alive, **kwargs)

        return pd.DataFrame(data=ltvs, columns=['ltv'])
