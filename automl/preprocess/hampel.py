import copy
import pandas as pd
import numpy as np
from typing import Union
from preprocess.base_prerocess import BasePrerocess
from models.base_model import BaseModel

class HampelFilter(BasePrerocess):
    def __init__(self, window: int = 5, sigma: float = 3, scale_factor: float = 1.4826, is_del_outlier: bool = False):
        self.window = window
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.is_del_outlier = is_del_outlier

    def fit(self, df: pd.DataFrame):
        pass

    def __hampel_filter(self, data: Union[list, pd.Series]):
        if isinstance(data, list):
            data = pd.Series(data)

        self.window = max(1, self.window)
        if self.window > len(data):
            self.window = len(data)
        if self.sigma <= 0:
            self.sigma = 3
        if self.scale_factor <= 0:
            self.scale_factor = 1.4826

        data = pd.concat([data.iloc[:self.window], data, data.iloc[-self.window:]])
        rolling_median = data.rolling(window=2 * self.window, center=True).median()
        delta = pd.Series.abs(data - rolling_median)
        rolling_mad = data.rolling(window=2 * self.window, center=True).apply(
            lambda x: pd.Series.median(pd.Series.abs(x - pd.Series.median(x)))
        )

        new_data = delta > self.sigma * self.scale_factor * rolling_mad
        new_data = new_data.iloc[self.window:-self.window]
        return new_data

    @BaseModel.check_columns_decorator(need_cols=[BaseModel.target_col_name])
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = copy.deepcopy(df)
        target = new_df[BaseModel.target_col_name]
        anomalies = self.__hampel_filter(target).astype(int)
        new_df['is_hampel_outlier'] = anomalies

        if self.is_del_outlier:
            new_df.loc[new_df['is_hampel_outlier'] == 1, BaseModel.target_col_name] = np.NaN
        
        new_df.drop(columns=['is_hampel_outlier'], inplace=True)
        return new_df

    def inverse_transform(self, df: pd.DataFrame, target_col=BaseModel.target_col_name) -> pd.DataFrame:
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        df = self.transform(df)
        return df