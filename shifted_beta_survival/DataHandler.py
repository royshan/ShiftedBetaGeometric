import numpy as np


class DataHandler(object):
    """
    DataHandler is an object to perform several manipulations to a pandas
    dataframe making it suitable to be fed to a ShiftedBeta object.

    Given a dataframe the user specifies an age and alive fields, and possible
    feature fields. These will be processed to comply with the input API of
    the lower lever ShiftedBeta object.

    Categorical features will be one-hot-encoded, a bias column can be added,
    and numerical features can also be normalized during the pre-processing
    stage.
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

        :param alive: str
            The column name with the status of each individual. In the context
            of survival analysis, an individual may be dead or alive, and its
            contribution to the model will depend on it.

        :param features: str, list or None
            A string with the name of the column to be used as features, or a
            list of names of columns to be used as features or None, if no
            features are to be used.

        :param bias: bool
            Whether or not a bias term should be added to the feature matrix.

        :param normalize: bool
            Whether or not numerical fields should be normalized (centered and
            scaled to have std=1)
        """

        # The name of age and alive fields
        self.age = age
        self.alive = alive

        # Make sure we have something to be used as a feature matrix. I.e.:
        # features=None and bias=False should raise an error.
        if features is None and not bias:
            raise ValueError("bias must be True if no features are being used "
                             "(features=None).")

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
        A method to sort features and divide them into categorical and
        numerical, and storing the names in sorted lists.

        Moreover this method also construct the dictionary feat_map, which
        takes categorical feature names as keys and a list with all the
        values they take as values.

        :param df: pandas DataFrame
            pandas dataframe from which features will be extracted

        :param features:
            list of columns names that will be used as features.

        :return: tuple (list, dict, list)
            A tuple with:
                A list of names of categorical features
                A dict with categorical feature name: all possible values
                A list of names of numerical features
        """

        # No features? No problem! Return empty object and let the code deal
        # with it further ahead.
        if features is None:
            return [], {}, []

        # Yes Features? Do stuff!!
        # Start by identifying which features are categorical by checking the
        # data types of columns in the dataframe. Those with data type
        # "category" are added to the list.
        cat_list = df.columns[df.dtypes == 'category']
        cat_list = [cat for cat in cat_list if cat in features]

        # Update them categorical features
        cat_list = sorted(cat_list)

        # Build feature maps!
        feat_map = {}

        # Loop over the list of categorical features creating a map that takes
        # each factor to an integer, in alphabetical order. So if feature
        # 'city' takes values 'NY', 'LA' and 'SF' the feature map will have the
        # entry: {'city': {'LA': 0, 'NY': 1, 'SF': 2}, ...}. Similarly for all
        # categorical features present.
        for feature in cat_list:
            feat_map[feature] = dict(zip(sorted(df[feature].cat.categories),
                                         range(len(df[feature].cat.categories))))

        # Update list of numerical features. Make it such any feature that was
        # given by the user and it is not a categorical feature, will be
        # considered a numerical feature.
        num = sorted([feat for feat in features if feat not in cat_list])

        # Returns both lists
        return cat_list, feat_map, num

    def _one_hot_encode(self, df, categoricals):
        """
        A method to one-hot-encode categorical features. It computes a dummy
        matrix for each categorical column and returns a concatenation of all
        such matrices.


        :param df: pandas DataFrame
            Pandas dataframe from which features are to be extracted.

        :param categoricals:
        :return:
        """
        # Make sure these are sorted so we don't get things mixed up later!
        categoricals = sorted(categoricals)

        # dict to hold matrices of OHE features
        ohed_map = {}

        # Loop over each categorical feature and create appropriate dummy
        # matrices. The size of the matrix is dictated by the feature_map
        # instance variable, that, for each categorical feature, hold a dict
        # with as many key: val pairs as there are factors in the training set
        for feature in categoricals:
            # categoricals: sorted list with names of categorical features
            #      feature: name of a categorical feature

            # Use the size of the dictionary to create a dummy matrix with the
            # appropriate size. ** Redundant since dummy matrix need
            # only to have n_factors - 1 columns, however this model only makes
            # sense with a bias variable added, which takes care of that.
            ohed_map[feature] = np.zeros((df.shape[0],
                                          len(self.feature_map[feature])),
                                         dtype=int)

        # Sometimes it is useful to know if new factors were found in the test
        # set, so we create a dict to store a list of new factors found in each
        # categorical feature.
        warning_new = {}

        # Mutable object to store index changes! No need for pandas index, phew!
        # using a mutable list seemed like the easiest answer, even if its a bit
        # ugly.
        currow_mutable = [0, ]

        # Internal function that is passed to pandas' apply method.
        def update_ohe_matrix(row, categorical_columns, currow):
            """
            Given a row of a pandas DataFrame, this function updates all
            dummy matrices corresponding to each categorical variables.

            Notice that the feature_map variable was generated in a such a way
            to immediately lends itself to populating an empty matrix in dummy
            fashion.

            So this function can simply use the value in the factor: value pair
            to set to one the correct column in the dummy matrix.

            As for rows, we use an external, mutable object to keep track of
            which row the pandas object is reading the data from and use that
            to update the correct row of the dummy matrices.

            :param row: row of a pandas DataFrame
                The row passed by the method apply of a pandas DataFrame.

            :param categorical_columns: list
                A list with the name of all categorical features that should be
                one hot encoded.

            :param currow: list (mutable object)
                An external, mutable object used to keep track of which row of
                the dataframe is being read (in case indexes are messed up).
            """

            # Loop over all categorical features
            for curr_feature in categorical_columns:
                # categorical_columns: list of names of categorical variables
                #        curr_feature: name of categorical variable

                # Read the index of current row from external mutable object
                row_index = currow[-1]

                # Value of current feature in current row
                row_feat_val = row[curr_feature]

                # Sometimes the test set contains factors that were not present
                # at training time, so we must take that into account. We do so
                # by catching a KeyError generated by trying to read an invalid
                # value off of a dictionary.
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

            # Update the index so that the next row with be read the next time
            # this function is invoked.
            currow[-1] += 1

        # apply function with pandas apply method
        df.apply(lambda row: update_ohe_matrix(row,
                                               categoricals,
                                               currow_mutable),
                 axis=1)

        # Print some warking in case new factors were found.
        if len(warning_new) > 0:
            print('WARNING: NEW STUFF: {}'.format(warning_new))

        # Return a matrix complized of all dummy matrices for each categorical
        # variables concatenated along columns.
        return np.concatenate([xohe for key, xohe in ohed_map.items()], axis=1)

    def fit(self, df):
        """
        This method is responsible for scanning the dataframe -df- and storing
        the necessary parameters and statistics used to pre-process all
        features.

        It starts by splitting the full features list into a categorical and a
        numerical list of features, while also creating the feature_map for
        categorical features. This is done by invoking the _get_categoricals
        method.

        That being done, the next step is computing the mean and standard
        deviations of all numerical features in case the normalize keyword was
        set to True when the object was created. These statistics are stored
        in a dictionary to be used at transformation time.

        Finally the method sets the fitted_model flag to True.

        :param df: pandas DataFrame
            The dataframe on which categorical and numerical features will be
            based.
        """

        # Get types of features (in place updates!)
        cat, feat_map, num = self._get_categoricals(df=df,
                                                    features=self.features)

        # store features
        self.categorical.extend(cat)
        self.feature_map = feat_map

        # numerical
        self.numerical.extend(num)
        # should we center and standard?
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
        should transform have the option of working with a dataframe missing
        the "target" variables (age and alive)? while these are not necessary
        for computing the survival and churn curves, they are used when making
        ltv predicitons.

        :param df:
        :return:
        """
        # write a better message!
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

        # When transforming the dataset, age and alive field must not be always
        # present. If that's the case, we return None. To make life easier, and
        # avoid an ugly chain of if statements, we create a nice little
        # function to handle it.
        def returner(data_frame, key):
            try:
                return data_frame[key].values
            except KeyError:
                return None

        return xout, returner(df, self.age), returner(df, self.alive)

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
            # Sort everything to avoid naming things incorrectly. Notice
            # that all names should be sorted in their origin. However,
            # it doesn't hurt to be extra safe..
            for cat_name in sorted(self.categorical):
                for category in sorted(self.feature_map[cat_name]):
                    names.append(cat_name + "_" + category)

        return names
