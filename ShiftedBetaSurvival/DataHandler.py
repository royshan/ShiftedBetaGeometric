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

    def __init__(self, cohort, age, category=None, types=None):
        """
        The object is initialized with the dataset to be transformed, the name
        of the fields identified as cohort, individual age and optional
        category(ies) to be used as predictors.

        :param cohort: str
            The column name to identify to which cohort each individual belongs
            to.

        :param age: str
            The column name to identify the age of each individual. Age has to
            be an integer value, and will determine the time intervals the
            model with work with.

        :param category: str or list of str
            A list or string with the column name(s) of fields to be used as
            features. These fields are treated as categorical variables and
            are one hot encoded and fed to a linear model.
        """

        self.cohort = cohort
        self.age = age

        # If the category name was passed as a single string, we turn it into
        # a list of one element (not list of characters, as you would get with
        # list('abc').
        if isinstance(category, str):
            category = [category]
        # Try to explicitly transform category to a list (perhaps it was passed
        # as a tuple or something. If it was None to begin with, we catch a
        # TypeError and move on.
        try:
            self.category = sorted(category)
        except TypeError:
            self.category = None

        if types is None:
            self._get_types()
        else:
            self.types = types

    def _get_types(self):
        self.types = 0
