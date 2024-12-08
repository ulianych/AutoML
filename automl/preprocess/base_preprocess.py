from abc import ABC, abstractmethod
from models.base_model import BaseModel
import pandas as pd
from typing import Union

class BasePreprocess(ABC):

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    @BaseModel.check_columns_decorator(need_cols=[BaseModel.target_col_name])
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass