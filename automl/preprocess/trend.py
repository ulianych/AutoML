import copy
import pandas as pd
import numpy as np
from typing import Union
from models.base_model import BaseModel
from preprocess.base_prerocess import BasePrerocess
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

class DetrendingData(BasePrerocess):
    def __init__(self, group_by_columns: list = [], type='poly', poly_degree=1):
        self.type = type
        self.poly_degree = poly_degree
        self.models = {}
        self._x_median = {}
        self.group_by_columns = [*group_by_columns]

    def _poly_features(self, df: pd.DataFrame, target_col_name=BaseModel.target_col_name):
        new_df = df.copy(deep=True)
        x = new_df[BaseModel.date_col_name]
        x = x.apply(lambda ts: ts.timestamp()).to_numpy().reshape(-1, 1)
        y = df[target_col_name].tolist()
        return x, y

    def _exp_features(self, df: pd.DataFrame):
        x = df[BaseModel.date_col_name].apply(lambda ts: ts.timestamp()).to_numpy().reshape(-1, 1)
        y = df[BaseModel.target_col_name].to_numpy()
        return x, y

    def _get_x_y(self, df: pd.DataFrame):
        if self.type == 'poly':
            x, y = self._poly_features(df)
        elif self.type == 'exp':
            x, y = self._exp_features(df)
        else:
            raise ValueError('No such type, try to use "poly" or "exp" instead.')
        return x, y

    def _remove_trend(self, df: pd.DataFrame, key):
        if not hasattr(self.models[key], 'regressor'):
            raise ValueError("Linear regression wasn't fitted. \nTry calling fit method.")
        
        x, y = self._get_x_y(df)
        x -= self._x_median[key]
        y_pred = self.models[key].predict(x)
        y_detrended = y - y_pred
        df[BaseModel.target_col_name] = y_detrended
        return df

    @BaseModel.check_columns_decorator(need_cols=[BaseModel.target_col_name])
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy(deep=True)
        new_df.sort_values(BaseModel.date_col_name, inplace=True)
        detrended_data = []
        grouped_data = new_df.groupby(self.group_by_columns) if self.group_by_columns else [('all', new_df)]

        for pair, group in grouped_data:
            temp = self._remove_trend(group, pair)
            detrended_data.append(temp)

        detrended_df = pd.concat(detrended_data, ignore_index=True)
        return detrended_df

    def fit(self, df: pd.DataFrame):
        grouped_data = df.groupby(self.group_by_columns) if self.group_by_columns else [('all', df)]
        
        for pair, group in grouped_data:
            if self.type == 'poly':
                self.models[pair] = Pipeline([
                    ("polynomial", PolynomialFeatures(degree=self.poly_degree, include_bias=False)),
                    ("regressor", LinearRegression())
                ])
            elif self.type == 'exp':
                self.models[pair] = Pipeline([
                    ("regressor", LinearRegression())
                ])
            
            x, y = self._get_x_y(group)
            self._x_median[pair] = np.median(x)
            x -= self._x_median[pair]
            self.models[pair].fit(x, y)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame, target_col=BaseModel.target_col_name) -> pd.DataFrame:
        new_df = df.copy(deep=True)
        new_df.sort_values(BaseModel.date_col_name, inplace=True)
        grouped_data = new_df.groupby(self.group_by_columns) if self.group_by_columns else [('all', new_df)]
        trended_data = []

        if self.type == 'poly':
            for pair, group in grouped_data:
                x, y = self._poly_features(group, target_colname=target_col)
                x -= self._x_median[pair]
                y_pred = self.models[pair].predict(x)
                group[target_col] = y + y_pred
                trended_data.append(group)

        elif self.type == 'exp':
            for pair, group in grouped_data:
                y = group[target_col]
                x = group[BaseModel.date_col_name].apply(lambda ts: ts.timestamp()).to_numpy().reshape(-1, 1)
                x -= self._x_median[pair]
                y_pred = self.models[pair].predict(x)
                y_pred = np.exp(y_pred)
                group[target_col] = y + y_pred
                trended_data.append(group)
        else:
            raise ValueError('No such type, try to use "poly" or "exp" instead.')

        return pd.concat(trended_data, ignore_index=True)