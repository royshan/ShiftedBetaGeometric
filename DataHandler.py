import numpy


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

    Additionally it can compute the number of idividuals lost per cohort
    returning a similar dictionary as above with key: number of lost
    individuals pairs.

    Moreover a method to zip population with number of lost individuals
    exist, which is precisely the format accepted by a shifted beta object.

    Finally it is also capable of padding smaller cohorts with zeros
    adjusting all lists of cohort population to have the same length.
    """

    def __init__(self, data, cohort, age, category=None):
        """
        The object is initialized with the dataset to be transformed, the name
        of the fields identified as cohort, individual age and optional
        category(ies) to be used as predictors.

        :param data: pandas DataFrame
            A system level pandas dataframe. Similar to:
            _______
            id | cohort | age | predictors...
            1  |      a |   3 | ...
            2  |      b |   7 | ...
            3  |      a |   4 | ...
            ...

        :param cohort: str
            The column name to identify to which cohort each individual belongs
            to.

        :param age: str
            The column name to identify the age of each individual. Age has to
            be an integer value, and will determine the time intervals the
            model with work with.

        :param category: list of str
            A list with the column name(s) of fields to be used as features.
            These fields are treated as categorical variables and are one hot
            encoded and fed to a linear model.
        """

        self.data = data
        self.cohort = cohort
        self.age = age

        # if cat is string, transform it...
        try:
            self.category = list(category)
        except TypeError:
            self.category = None

    def paired_data(self):
        pass

    def aggregate(self):
        """
        A method to turn a system level data set into lists of cohort
        population, as requires by the shifted beta model.

        Given a pandas dataframe of the kind:

        _______
        id | cohort | age | predictors...
        1  |      a |   3 | ...
        2  |      b |   7 | ...
        3  |      a |   4 | ...
        ...

        This method aggregates the data and return the number of live ids at a
        given time period per cohort. The aggregation can be done globally,
        only discriminating the cohorts, or it can be done by predictor (which
        are assumed to be categorical).

        :return: dict
            A dictionary with predictor: list of cohort population pairs. If no
            category is passed (category = None), then the aggregation is done
            globally, and the key is simply 'data'.
        """

        # If category is None, no differentiation is done other than
        # cohort-wise. The output is still a dictionary, but the key is a
        # generic key ('data').#
        if self.category is None:
            # Initiate the output dictionary with a generic key and an empty
            # list to hold the lists of populations.
            out = {'data': []}

            # Call to the aggregator method. It alters the output dict in
            # place.
            self.aggregator(out, 'data', self.data, self.cohort, self.age)
        else:
            # Initiate an empty dictionary. Key will be entered as needed.
            out = {}

            # Loop through all columns names that will be used as categories
            for category in self.category:

                # Split data by category with pandas group by.
                for k, kdf in self.data.groupby(category):

                    # The key name is given by the category column and the
                    # category value, separated by underscore.
                    cur_name = category + '_' + str(k)

                    # Check if the current name is already an existing key,
                    # if it is not, initialize it with an empty list.
                    if cur_name not in out:
                        out[cur_name] = []

                    # Call to the aggregator method. It alters the output dict
                    # in place.
                    self.aggregator(out, cur_name, kdf, self.cohort, self.age)

        return out

    @staticmethod
    def aggregator(out_dict, out_name, kind_data, cohort_field, age_field):
        """
        aggregator is a static method to alter the output data dictionary in
        place. It appends a list with the cohort population at for every
        possible age value.

        :param out_dict:
        :param out_name:
        :param kind_data:
        :param cohort_field:
        :param age_field:
        :return:
        """

        for name, df in kind_data.groupby(cohort_field):

            cdata = numpy.zeros(df[age_field].max(), dtype=int)

            for index, row in df.iterrows():

                cdata[:row[age_field]] += 1

            out_dict[out_name].append(list(cdata))

    @staticmethod
    def n_lost(data):
        """
        A static method to compute the number of lost customers lost at each
        time period given a list of total number of customers over time. It is
        used to compute the likelihood.

        :param data: List
            A list with the number of cohort customers still active
            at a given moment in time.

        :return: List
            A list with the number of customers lost at a given moment in time

        """

        # i-th element in this list will hold the number of people that left
        # from months i-1 and i. Its 0-th element is then initialized to None
        # and all subsequent ones calculated from the number of active
        # customers as months i and i-1 and appended to the list below.
        # Finally, the method returns the list.
        lost_dict = {key: [] for key in data}

        for key, cohorts in data.iteritems():

            for cohort in cohorts:
                # Initialize with None and calculate subsequent months.
                lost_num = [None]
                for i in range(1, len(cohort)):
                    lost_num.append(cohort[i - 1] - cohort[i])

                lost_dict[key].append(lost_num)

        return lost_dict

    @staticmethod
    def get_sizes(data):
        """
        This method finds the sizes of all cohorts present in the dataset. And
        returns a list with the sizes (number of months in record) for each of
        the cohorts present in the dataset.

        :param data: List
            A list of lists where each element is a list containing a cohort's
            population evolution over time.

        :return: List of integers
            A list with the sizes (number of months in record) for each of
            the cohorts present in the dataset.
        """

        sizes = []

        for cohort in data:
            sizes.append(len(cohort))

        return sizes

    @staticmethod
    def extend_data(data, size):
        """
        Cohort data will generally have different sizes (the number of months
        since upgrading). However, the mcmc object is easier to handle if
        similarly shaped data is passed to it. Therefore we augmented all
        cohorts but the oldest (largest) one by appending zeros to the end.

        ** Further in the model these zeros will be identified and correctly
        accounted for (ignored).**

        :param data: List
            A list of lists where each element is a list containing a cohort's
            population evolution over time.

        :param size: Integer
            The size of the longest list in data.

        :return: List
            The original dataset with all but the longest list augmented with
            zeros.
        """

        # Make sure the cohort data has at least two entries, otherwise it
        # makes little sense to talk about the number of customers lost!
        # Raise assertion error if condition are not met.
        try:
            assert len(data) > 1
        except AssertionError as err:
            err.args += ('Cohort has insufficient data, at least two months'
                         'are necessary.', )
            raise

        extended_data = list(data)

        for cohort in extended_data:
            # Add enough zeros to make the size of the current list match that
            # of the longest one. Skip if cohort in question is the oldest
            # available.#
            if len(cohort) < size:
                cohort += [0] * (size - len(cohort))

        return extended_data
