from datetime import timedelta
import numpy as np
import pandas as pd
from multipledispatch import dispatch
from typing import List
from models.base_model import BaseModel
from models.catboost_model import CatBoostModel
from models.base_optuna import AutoOptuna
from preprocess import BasePrerocess
import time

class AutoML:
    def __init__(self,
                 groupby_column=None,
                 model: BaseModel = None,
                 list_of_preprocess: List[BasePrerocess] = None,
                 count_of_date: int = 7):
        self.list_of_preprocess = list_of_preprocess
        self.model = model or CatBoostModel()
        self.count_of_date = count_of_date
        self.groupby_column = groupby_column
        self.total_score = 1

    def preprocess(self, df: pd.DataFrame, inverse=False, target_col=BaseModel.target_col_name) -> pd.DataFrame:
        local_df = df.copy(deep=True)
        for process in self.list_of_preprocess:
            print(process)
            if inverse:
                local_df = process.inverse_transform(local_df, target_col=target_col)
            else:
                local_df = process.fit_transform(local_df)
            print(local_df[local_df['clientbasenumber'] == '100031'])
        return local_df

    def train(self, df: pd.DataFrame, test_df: pd.DataFrame = None):
        if isinstance(self.model, AutoOptuna) and test_df is not None:
            self.model.get_objective_func(df, len(test_df[BaseModel.date_col_name].unique()))
        self.model.fit(df)

    def predict(self, df: pd.DataFrame) -> (float, pd.DataFrame):
        local_df = df.copy(deep=True)
        return self.model.predict(local_df)

    @dispatch(pd.DataFrame, pd.DataFrame)
    def execute(self, train_df: pd.DataFrame, test_df: pd.DataFrame, is_train=True) -> pd.DataFrame:
        result_df = None
        start_test_day = test_df[BaseModel.date_col_name].min()
        total_df = pd.concat([train_df, test_df]).sort_values(by=[BaseModel.date_col_name], ascending=False)
        print(total_df[total_df['clientbasenumber'] == '100031'])

        l_df_seg = list(zip(*total_df.groupby(self.groupby_column)))[1] if self.groupby_column else [total_df]
        
        for l_df in l_df_seg:
            l_df = self.preprocess(l_df)
            print(l_df[l_df['clientbasenumber'] == '100031'])
            l_train_df, l_test_df = BaseModel.train_test_split(l_df, start_test_day=start_test_day)
            print(l_test_df)
            if is_train:
                self.train(l_train_df, test_df=l_test_df)
            print(l_test_df.dtypes)

            l_test_df = self.predict(l_test_df.drop(columns=[BaseModel.target_col_name]))
            l_test_df = self.preprocess(l_test_df, inverse=True, target_col=BaseModel.prediction_col_name)
            result_df = pd.concat([result_df, l_test_df], ignore_index=True)

        # result_df.to_csv('/preprocess_df.csv', index=False)
        return result_df

    @dispatch(pd.DataFrame, pred_count_days=int, add_test_data=bool, groupby_column=List[str])
    def execute(self, df: pd.DataFrame, pred_count_days=1, add_test_data=True, groupby_column=[BaseModel.date_col_name]) -> pd.DataFrame:
        local_df = df.copy(deep=True)
        if add_test_data:
            local_df = self.create_test_data(local_df, groupby_column, pred_count_days)
        start_test_day = np.datetime64(local_df[BaseModel.date_col_name].max()) - timedelta(days=pred_count_days - 1)
        train_df, test_df = BaseModel.train_test_split(local_df, start_test_day=start_test_day)
        return self.execute(train_df, test_df)

    @BaseModel.check_columns_decorator()
    def create_test_data(self, df: pd.DataFrame, groupby_column, count_days=1) -> pd.DataFrame:
        if BaseModel.date_col_name not in groupby_column:
            groupby_column.append(BaseModel.date_col_name)
        local_df = df.drop_duplicates(subset=groupby_column).reset_index(drop=True)

        start_day = local_df[BaseModel.date_col_name].max() - timedelta(days=30)
        max_day = local_df[local_df[BaseModel.date_col_name] >= start_day].groupby(BaseModel.date_col_name).agg('count')[BaseModel.target_col_name].idxmax()
        test_df = local_df[local_df[BaseModel.date_col_name] == max_day].reset_index(drop=True)

        date = np.repeat([local_df[BaseModel.date_col_name].max() + timedelta(days=i + 1) for i in range(count_days)], len(test_df))
        test_df = test_df.loc[np.tile(test_df.index, count_days)].reset_index(drop=True)
        test_df[BaseModel.date_col_name] = date
        test_df[test_df.columns.drop(groupby_column)] = np.NaN

        local_df = pd.concat([local_df, test_df])
        return local_df