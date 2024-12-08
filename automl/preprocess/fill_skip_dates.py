import copy
import pandas as pd
from models import BaseModel
from preprocess.base_prerocess import BasePrerocess
from preprocess.fillna_preprocess import FillnaPreprocess

class FillSkipDates(BasePrerocess):
    def __init__(self, freq: str = 'D', group_by_columns: list = []):
        self.freq = freq
        self.group_by_columns = [*group_by_columns]

    @BaseModel.check_columns_decorator(need_cols=[BaseModel.target_col_name])
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = copy.deepcopy(df)
        df_arr = []

        unique_pairs = new_df.groupby(self.group_by_columns) if self.group_by_columns else [('all', new_df)]
        for pair, data in unique_pairs:
            new_dates = pd.date_range(start=data[BaseModel.date_col_name].min(), 
                                       end=data[BaseModel.date_col_name].max(), 
                                       freq=self.freq)
            new_data = pd.DataFrame({BaseModel.date_col_name: new_dates})
            data = pd.merge(new_data, data, on=BaseModel.date_col_name, how='left')
            if pair != 'all':
                data[self.group_by_columns] = pair
            df_arr.append(data)

        final_df = pd.concat(df_arr, ignore_index=True)
        return final_df

    def fit(self, df: pd.DataFrame):
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        new_df = self.transform(df)
        return new_df

    def inverse_transform(self, df: pd.DataFrame, target_col=BaseModel.target_col_name) -> pd.DataFrame:
        return df