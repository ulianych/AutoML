import pandas as pd
from abc import ABC, abstractmethod
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import datetime

class BaseModel(ABC):
    __TARGET_COL_NAME = 'target'
    __DATE_COL_NAME = 'daytime'
    __PREDDICTION_COL_NAME = 'prediction'

    @classmethod
    @property
    def target_col_name(cls):
        return cls.__TARGET_COL_NAME

    @classmethod
    @property
    def date_col_name(cls):
        return cls.__DATE_COL_NAME 
    
    @classmethod
    @property
    def prediction_col_name(cls):
        return cls.__PREDDICTION_COL_NAME

    @abstractmethod
    def fit(self, df: pd.DataFrame) :
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) :
        pass

    @abstractmethod
    def score(self, df: pd.DataFrame) :
        pass

    @classmethod
    def check_columns(cls, df: pd.DataFrame, need_cols: list = None):
        columns = df.columns.tolist()
        need_cols = need_cols or (cls.__TARGET_COL_NAME, cls.__DATE_COL_NAME)
        for name in need_cols:
            if name not in columns:
                raise Exception(f'"{name}" column is missing')
            
        if cls.__DATE_COL_NAME in need_cols and not is_datetime(df[cls.__DATE_COL_NAME]):
            raise Exception(f'column {cls.__DATE_COL_NAME} must be datetime64[ns] type')
        
    @classmethod
    def check_columns_decorator(cls, need_cols=None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                df = kwargs.get('df') if 'df' in kwargs else args[1]
                BaseModel.check_columns(df, need_cols=need_cols)
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    

    @classmethod
    def train_test_split(cls, df: pd.DataFrame,
                         test_size: float = 0.2,
                         start_test_day=None,
                         count_of_days=None) -> (pd.DataFrame, pd.DataFrame):
        cls.check_columns(df)

        start_date = df[cls.__DATE_COL_NAME].min()
        end_date = df[cls.__DATE_COL_NAME].max()
        first_day_test = (end_date - datetime.timedelta(days=count_of_days - 1)) if count_of_days is not None else None
        first_day_test = first_day_test or (
            end_date - datetime.timedelta(days=int((end_date - start_date).days * test_size) - 1))
        start_test_day = start_test_day or first_day_test
        train_df = df[df[cls.__DATE_COL_NAME] < start_test_day]
        test_df = df[df[cls.__DATE_COL_NAME] >= start_test_day]

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

