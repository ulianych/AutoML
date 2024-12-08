import pandas as pd
from main import AutoML
from preprocess import HampelFilter, LagPreprocess, RomanovskyFilter
from preprocess.fill_skip_dates import FillSkipDates
from preprocess.feature_trend_columns import FeatureTrend
from preprocess.fillna_preprocess import FillnaPreprocess
from preprocess.new_feature_added import NewFeature
from preprocess.trend import DetrendingData
from preprocess.calender import CalenderFilter
from models import CatBoostModel
from models.mstl_model import MSTLmodel
from models.arima_model import ARIMAModel
from models.random_forest import RandomForest
from models.base_model import BaseModel
from sklearn.metrics import mean_absolute_error as mae

class OrchestraModel(BaseModel):
    def __init__(self, models=None, agg_model=None, group_by_columns=None):
        self.group_by_columns = [] if group_by_columns is None else [*group_by_columns]
        self.agg_model = RandomForest(drop_col_name=self.group_by_columns) if agg_model is None else agg_model
        
        if models is not None:
            self.models = models
        else:
            self.models = [
                CatBoostModel(cat_col_name=[], drop_col_name=self.group_by_columns),
                MSTLmodel(group_by_columns=self.group_by_columns),
                RandomForest(drop_col_name=self.group_by_columns)
            ]

    @BaseModel.check_columns_decorator()
    def fit(self, df: pd.DataFrame):
        col_name = [*self.group_by_columns]
        
        if self.date_col_name not in col_name:
            col_name.append(self.date_col_name)
        if self.target_col_name not in col_name:
            col_name.append(self.target_col_name)

        df4train = df[col_name]
        
        for model in self.models:
            model.fit(df)
            cur_df = model.predict(df).sort_values(by=[self.date_col_name])
            cur_df = cur_df[[self.prediction_col_name, self.date_col_name, *self.group_by_columns]]
            df4train = df4train.merge(cur_df, on=[self.date_col_name, *self.group_by_columns])
            df4train.rename(columns={self.prediction_col_name: type(model).__name__}, inplace=True)

        self.agg_model.fit(df4train)

    @BaseModel.check_columns_decorator(need_cols=[BaseModel.date_col_name])
    def predict(self, df: pd.DataFrame, count_days4pred: int = 7):
        df_local = df.copy(deep=True)
        col_name = [*self.group_by_columns]

        if self.date_col_name not in col_name:
            col_name.append(self.date_col_name)

        df4train = df_local[col_name]
        
        for model in self.models:
            cur_df = model.predict(df_local).sort_values(by=[self.date_col_name])
            cur_df = cur_df[[self.prediction_col_name, self.date_col_name, *self.group_by_columns]]
            df4train = df4train.merge(cur_df, on=[self.date_col_name, *self.group_by_columns])
            df4train.rename(columns={self.prediction_col_name: type(model).__name__}, inplace=True)

        df_local[self.prediction_col_name] = self.agg_model.predict(df4train)[self.prediction_col_name]
        return df_local

    def score(self, y_true, y_pred) -> float:
        pass