from __future__ import print_function
from DataHandler import DataHandler
from ShiftedBeta import ShiftedBeta
import numpy as np
import pandas as pd


class ShiftedBetaSurvival(object):
    """
    This class implements an extended version of the Shifted-Beta-Geometric
    model by P. Fader and B. Hardie.

    The original model works by assuming a constant in time, beta distributed
    individual probability of churn. Due to the heterogeneity of a cohort's
    churn rates (since each individual will have a different probability of
    churning), expected behaviours such as the decrease of cohort churn rate
    over time arise naturally.

    The extension done here generalizes the coefficients alpha and beta of the
    original model to function of features on the individual level. A
    log-linear model is used to construct alpha(x) and beta(x) and the
    likelihood is then computed by combining the contributions of each and
    every sample in the training set.

    The model takes as inputs ...
    """

    def __init__(self,
                 age,
                 alive,
                 features=None,
                 gamma=1.0,
                 gamma_beta=None,
                 bias=True,
                 normalize=True,
                 verbose=False):
        """
        Initializes objects with parameters necessary to create the supporting
        objects: DataHandler and ShiftedBeta

        :param age: str
            The column name to identify the age of each individual. Age has to
            be an integer value, and will determine the time intervals the
            model with work with.
                --- See DataHandler.py

        :param alive: str
            The column name with the status of each individual. In the context
            of survival analysis, an individual may be dead or alive, and its
            contribution to the model will depend on it.
                --- See DataHandler.py

        :param features: str, list or None
            A string with the name of the column to be used as features, or a
            list of names of columns to be used as features or None, if no
            features are to be used.
                --- See DataHandler.py

        :param gamma: float
            A non-negative float specifying the strength of the regularization
            applied to w_alpha (alpha's weights) and, if gamma_beta is not
            given, it is also applied to beta.
                --- See ShiftedBeta.py

        :param gamma_beta: float
            A non-negative float specifying the strength of the regularization
            applied to w_beta (beta's weights). If specified, overwrites the
            value of gamma for beta.
                --- See ShiftedBeta.py

        :param bias: bool
            Whether or not a bias term should be added to the feature matrix.
                --- See DataHandler.py

        :param normalize: bool
            Whether or not numerical fields should be normalized (centered and
            scaled to have std=1)
                --- See DataHandler.py

        :param verbose: bool
            Whether of not status updates should be printed
                --- See ShiftedBeta.py
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
        # Was a different gammab parameter passed?
        if gamma_beta is None:
            gamma_beta = 1.0 * gamma
        # create shifted beta object
        self.sb = ShiftedBeta(gamma_alpha=gamma,
                              gamma_beta=gamma_beta,
                              verbose=verbose)

    def fit(self, df, restarts=1):
        """
        A method responsible for learning both the transformation of the data,
        including addition of a bias parameters, centering and re-scaling of
        numerical features, and one-hot-encoding of categorical features. In
        addition to learning the parameters alpha and beta of the shifted-beta-
        geometric model.

        This is just a wrapper, the real heavy-lifting is done by the
        DataHandler and ShiftedBeta objects.

        :param df: pandas DataFrame
            A pandas DataFrame with similar schema as the one used to train
            the model. Similar in the sense that the columns used as cohort,
            age and categories must match. Extra columns with not affect
            anything.

        :param restarts: int
            Number of times to restart the optimization procedure with a
            different seed, to avoid getting stuck on local maxima.
        """
        # Transform dataframe extracting feature matrix, ages and alive status.
        x, y, z = self.dh.fit_transform(df)

        # fit to data using the ShiftedBeta object.
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
