import numpy


class DataHandler:

    def __init__(self, data, cohort, age, kind=None):

        self.data = data
        self.cohort = cohort
        self.age = age
        self.kind = kind

    def aggregate(self):

        if self.kind is None:
            out = {'data': []}

            self.aggregator(out, 'data', self.data, self.cohort, self.age)
        else:
            out = {}

            for k, kdf in self.data.groupby(self.kind):

                cur_name = self.kind + '_' + str(k)

                if k not in out:
                    out[cur_name] = []

                self.aggregator(out, cur_name, kdf, self.cohort, self.age)

        return out

    @staticmethod
    def aggregator(out_dict, out_name, kind_data, cohort_field, age_field):

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
