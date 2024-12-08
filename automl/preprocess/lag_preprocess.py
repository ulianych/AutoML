import copy
import pandas as pd
from models import BaseModel
from preprocess.base_prerocess import BasePrerocess

class LagPreprocess(BasePrerocess):
    def __init__(self, days: list, group_by_columns: list = []):
        self.days = days
        self.group_by_columns = [*group_by_columns]

    @BaseModel.check_columns_decorator(need_cols=[BaseModel.target_col_name])
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = copy.deepcopy(df)
        new_df = new_df.sort_values(BaseModel.date_col_name)
        
        final_df = pd.concat(
            [new_df, new_df.groupby(self.group_by_columns)[BaseModel.target_col_name].shift(periods=self.days)],
            axis=1
        )
        return final_df

    def fit(self, df: pd.DataFrame):
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        new_df = self.transform(df)
        return new_df

    def inverse_transform(self, df: pd.DataFrame, target_col=BaseModel.target_col_name) -> pd.DataFrame:
        return df