from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd



class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        """
        :param columns: List of column names in X to select
        """
        self.columns = columns
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: pd.DataFrame
        :return: pd.DataFrame of selected Columns
        """
        assert isinstance(X, pd.DataFrame)

        try:
            return X.loc[:, self.columns]

        except KeyError:
            unknown_columns = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % unknown_columns)


class TypeSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that returns dataframe of specified dtype
    """
    def __init__(self, dtype=None):
        """
        :param dtype: one of 'object', 'bool' or 'number'
        """
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])



class ColumnExcluder(BaseEstimator, TransformerMixin):
    """
    Transformer that excludes specified columns
    """
    def __init__(self, cols_exclude=None):
        self.cols_exclude = cols_exclude

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert self.cols_exclude
        assert isinstance(X, pd.DataFrame)

        if isinstance(self.cols_exclude, str):
            self.cols_exclude = [self.cols_exclude]

        assert set(self.cols_exclude).issubset(X.columns)

        return X.loc[:, ~X.columns.isin(self.cols_exclude)]