from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, cols_to_drop):

        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X.drop(columns=self.cols_to_drop, errors='ignore')


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variable:str):
      if not isinstance(variable,str):
        raise ValueError("variable should be a list")
      self.variable = variable


    def fit(self, X:pd.DataFrame,y:pd.Series=None):
      self.encoder = OneHotEncoder(sparse_output=False)
      self.encoder.fit(X[[self.variable]])
      return self

    def transform(self, X:pd.DataFrame):
      df=X.copy()
      # encoder = OneHotEncoder(sparse_output=False)
      # encoder.fit(X[[self.variable]])
      encoded_weekday = self.encoder.transform(X[[self.variable]])
      enc_wkday_features = self.encoder.get_feature_names_out([self.variable])
      df[enc_wkday_features] = encoded_weekday
      print(f"One hot encoding----{df.head(2)}")
      return df

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable:str):
      if not isinstance(variable,str):
        raise ValueError("variable should be a list")
      self.variable = variable

    def fit(self, X:pd.DataFrame,y:pd.Series=None):
      return self

    def transform(self, X:pd.DataFrame):
      df = X.copy()
      q1 = df.describe()[self.variable].loc['25%']
      q3 = df.describe()[self.variable].loc['75%']
      iqr = q3 - q1
      lower_bound = q1 - (1.5 * iqr)
      upper_bound = q3 + (1.5 * iqr)
      for i in df.index:
        if df.loc[i,self.variable] > upper_bound:
            df.loc[i,self.variable]= upper_bound
        if df.loc[i,self.variable] < lower_bound:
            df.loc[i,self.variable]= lower_bound
      print(f"Outlier handling---handled for variable---{self.variable}")
      return df



class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        #for feature in self.variables:
        print(f"Beginning Mapper---{self.variables}")
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)
        # X[self.variables] = X[self.variables].astype(int)
        print(f"End Mapper---{self.variables}")
        return X


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self,variable:str):
      if not isinstance(variable,str):
        raise ValueError("variable should be a list")
      self.variable = variable

    def fit(self, X: pd.DataFrame, y:pd.Series=None):
      self.fill_value=X[self.variable].mode()[0]
      return self

    def transform(self, X: pd.DataFrame):
      X = X.copy()
      X[self.variable]=X[self.variable].fillna( self.fill_value)
      print(X[self.variable].isna().sum())

      return X


class WeekdayImputer(BaseEstimator, TransformerMixin):
  """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """
  def __init__(self,variable:str):
    if not isinstance(variable,str):
      raise ValueError("variable should be a list")
    self.variable = variable

  def fit(self,X:pd.DataFrame, y:pd.Series=None):
    # self.wkday_nullindex = X[X[self.variable].isnull() == True].index
    return self

  def transform(self,X:pd.DataFrame)->pd.DataFrame:
    df = X.copy()
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    self.wkday_nullindex = X[X[self.variable].isnull() == True].index
    df.loc[self.wkday_nullindex, self.variable] = df.loc[self.wkday_nullindex, 'dteday'].dt.day_name().apply(lambda x: x[:3])
    # print(df[self.variable].isna().sum())
    return df
