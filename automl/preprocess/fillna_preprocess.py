import copy
import pandas as pd
import numpy as np
from models import BaseModel
from preprocess.base_prerocess import BasePrerocess

class FillnaPreprocess(BasePrerocess):
    def __init__(self,
                 window: int = 3,
                 min_periods: int = 3,
                 group_by_columns: list = [],
                 col_types: pd.Series = None):
        self.window = window
        self.min_periods = min_periods
        self.group_by_columns = [*group_by_columns]
        self.col_types = col_types

    def __window_replace_value(self, df: pd.DataFrame):
        rolling_df = df.copy(deep=True)
        # Get line with not null value
        first_non_null = rolling_df.select_dtypes(include=np.number).apply(lambda col: col[col.first_valid_index()])
        first_line = rolling_df.iloc[0]
        first_line[first_non_null.index] = first_non_null
        first_line_df = first_line.to_frame().transpose()
        first_line_df = first_line_df.loc[np.repeat(first_line_df.index, self.min_periods)].reset_index(drop=True)

        print(first_line_df)
        rolling_df = pd.concat([first_line_df, rolling_df]).reset_index(drop=True)
        max_count_iter = 10 ** 4

        while rolling_df.select_dtypes(include=np.number).isna().values.any() and max_count_iter > 0:
            print(max_count_iter)
            max_count_iter -= 1
            rolling_df.fillna(
                rolling_df.select_dtypes(include=np.number)
                .rolling(self.window, self.min_periods, center=False, closed='both').mean(), inplace=True
            )
            rolling_df = rolling_df.iloc[self.min_periods:]

        return rolling_df

    def __set_raw_df_type(self, final_df: pd.DataFrame):
        if self.col_types is not None:
            same_keys = final_df.dtypes.to_dict().keys() & self.col_types.to_dict().keys()
            raw_type = dict(zip(same_keys, map(self.col_types.to_dict().get, same_keys)))
            new_type_col = pd.Series(final_df.dtypes.to_dict()).combine_first(raw_type)
            final_df = final_df.astype(new_type_col)

        return final_df

    @BaseModel.check_columns_decorator(need_cols=[BaseModel.target_col_name])
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = copy.deepcopy(df)
        new_df = new_df.sort_values(BaseModel.date_col_name)
        new_df.dropna(how='all', axis=1, inplace=True)
        col_one_value = new_df[new_df.columns.drop(self.group_by_columns)].nunique() < 2
        new_df.drop(columns=col_one_value[col_one_value].index, inplace=True)

        unique_pairs = new_df.groupby(self.group_by_columns) if self.group_by_columns else [('all', new_df)]
        final_df = pd.DataFrame()

        for pair, data in unique_pairs:
            final_df = pd.concat([final_df, self.__window_replace_value(data)])

        return self.__set_raw_df_type(final_df)

    def fit(self, df: pd.DataFrame):
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = self.transform(df)
        return new_df

    def inverse_transform(self, df: pd.DataFrame, target_col=BaseModel.target_col_name) -> pd.DataFrame:
        return df