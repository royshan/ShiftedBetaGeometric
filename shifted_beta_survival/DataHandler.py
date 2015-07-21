import numpy as np


class DataHandler(object):
    """
    DataHandler is an object to perform several manipulations to a pandas
    dataframe making it suitable to be fed to a ShiftedBeta object.

    Given a pandas dataframe of the kind:
    _______
    id | cohort | age | predictors...
    1  |      a |   3 | ...
    2  |      b |   7 | ...
    3  |      a |   4 | ...
    ...

    DataHandler turns it into a dictionary with key: list of cohort
    populations, key-value pairs.

    Additionally it can compute the number of individuals lost per cohort
    returning a similar dictionary as above with key: number of lost
    individuals pairs.

    Moreover a method to zip population with number of lost individuals
    exist, which is precisely the format accepted by a shifted beta object.

    Finally it is also capable of padding smaller cohorts with zeros
    adjusting all lists of cohort population to have the same length.
    """

    def __init__(self, age, alive, features=None, bias=True, normalize=True):
        """
        The object is initialized with the dataset to be transformed, the name
        of the fields identified as cohort, individual age and optional
        category(ies) to be used as predictors.

        :param age: str
            The column name to identify the age of each individual. Age has to
            be an integer value, and will determine the time intervals the
            model with work with.

        :param alive:
        :param features:
        :param bias:
        :param normalize:
        :return:
        """

        # Age and alive fields
        self.age = age
        self.alive = alive

        # If the category name was passed as a single string, we turn it into
        # a list of one element (not list of characters, as you would get with
        # list('abc').
        if isinstance(features, str):
            features = [features]
        # Try to explicitly transform category to a list (perhaps it was passed
        # as a tuple or something. If it was None to begin with, we catch a
        # TypeError and move on.
        try:
            self.features = sorted(features)
        except TypeError:
            self.features = None

        # What features are categorical?
        self.categorical = []
        self.numerical = []

        # OHE feature map to be constructed
        self.feature_map = {}

        # should bias be added
        self.add_bias = bias

        # standarize features?
        self.normalize = normalize
        self.stats = {'mean': {}, 'std': {}}

        # fit before transform!
        self.fitted_model = False

    @staticmethod
    def _get_categoricals(df, features):
        """

        :param df:
        :param features:
        :return:
        """

        # No features? No problem!
        if features is None:
            return [], {}, []

        # Yes Features? Do stuff!!
        cat_list = df.columns[df.dtypes == 'category']
        cat_list = [cat for cat in cat_list if cat in features]

        # Update them categorical features
        cat = sorted(cat_list)

        # Build feature maps!
        feat_map = {}

        for feature in cat:
            feat_map[feature] = dict(zip(sorted(df[feature].cat.categories),
                                         range(len(df[feature].cat.categories))))

        # Update numerical features
        num = sorted([feat for feat in features if feat not in cat])

        # Returns both lists
        return cat, feat_map, num

    def _one_hot_encode(self, df, categoricals):
        """

        :param df:
        :param categoricals:
        :return:
        """
        # Make sure these are sorted so we don't get things mixed up later!
        categoricals = sorted(categoricals)

        # dict to hold matrices of OHE features
        ohed_map = {}

        # Loop over each categorical feature and create appropriate feature
        # maps and what not
        for feature in categoricals:
            # categoricals:
            #      feature:
            ohed_map[feature] = np.zeros((df.shape[0],
                                          len(self.feature_map[feature])),
                                         dtype=int)

        warning_new = {}

        # Internal function that is passed to pandas' apply method.
        def update_ohe_matrix(row, categorical_columns):

            for curr_feature in categorical_columns:
                # categoricals:
                # curr_feature:

                # Index of current row
                row_index = row.name

                # Value of current feature in current row
                row_feat_val = row[curr_feature]

                try:
                    # Map between current categorical row-feature value and its
                    # numerical representation
                    mapped_val = self.feature_map[curr_feature][row_feat_val]

                    # Update OHE matrix by adding one to the appropriate column
                    # in the current row
                    ohed_map[curr_feature][row_index, mapped_val] += 1
                except KeyError:
                    try:
                        # Add newly seen value to warning dict
                        warning_new[curr_feature].add(row_feat_val)
                    except KeyError:
                        # If warning dict hasn't been populated yet,
                        # we do it here.
                        warning_new[curr_feature] = {row_feat_val}

        df.apply(lambda row: update_ohe_matrix(row, categoricals),
                 axis=1)

        if len(warning_new) > 0:
            print('WARNING: NEW STUFF: {}'.format(warning_new))

        return np.concatenate([xohe for key, xohe in ohed_map.items()], axis=1)

    def fit(self, df):
        """

        :param df:
        :return:
        """

        # Get types of features (in place updates!)
        cat, feat_map, num = self._get_categoricals(df=df, features=self.features)

        # store features
        self.categorical.extend(cat)
        self.feature_map = feat_map

        # numerical
        self.numerical.extend(num)
        # should be center and standard?
        if self.normalize and len(self.numerical) > 0:
            # pandas is awesome!
            stats = df[self.numerical].describe().T.to_dict()
            # update mean and std at once =)
            self.stats['mean'].update(stats['mean'])
            self.stats['std'].update(stats['std'])

        # update fitted status
        self.fitted_model = True

    def transform(self, df):
        """

        :param df:
        :return:
        """
        if not self.fitted_model:
            raise RuntimeError("Fit to data before transforming it.")

        if self.add_bias:
            xout = np.ones((df.shape[0], 1), dtype=int)
        else:
            xout = None

        # do we have numerical stuff?
        if len(self.numerical) > 0:

            # Numerical variables!
            num_vals = df[sorted(self.numerical)].values

            if self.normalize:
                for col, num_feat in enumerate(sorted(self.numerical)):
                    #    col:
                    # n_feat:
                    num_vals[:, col] -= self.stats['mean'][num_feat]
                    # Some arbitrary clip on minimum STD lest things break
                    num_vals[:, col] /= max(self.stats['std'][num_feat], 1e-4)

            # bias?
            if self.add_bias:
                xout = np.concatenate((xout, num_vals),
                                      axis=1)
            else:
                xout = num_vals

        if len(self.categorical) > 0:

            # get one hot encoded guys, sort categoricals just to be extra
            # sure (they are already sorted down the line).
            xohe = self._one_hot_encode(df, sorted(self.categorical))

            # bias?
            if xout is not None:
                xout = np.concatenate((xout, xohe),
                                      axis=1)
            else:
                xout = xohe

        if xout is None:
            raise ValueError('No data!')

        return xout, df[self.age].values, df[self.alive].values

    def fit_transform(self, df):
        """

        :param df:
        :return:
        """

        self.fit(df)
        return self.transform(df)

    def get_names(self):
        """
        A handy function to return the names of all variables in the
        transformed version of the dataset in the correct order. Particularly
        useful for the shiftedbetasurvival wrapper.

        :return: list
            list of names in correct order
        """
        names = []

        if self.add_bias:
            names.append('bias')

        if len(self.numerical) > 0:
            names.extend(sorted(self.numerical))

        if len(self.categorical) > 0:
            for cat_name in sorted(self.categorical):
                for category in sorted(self.feature_map[cat_name]):
                    names.append(cat_name + "_" + category)

        return names
