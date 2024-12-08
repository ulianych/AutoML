import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from catboost import Pool
from sklearn.metrics import mean_squared_error


from models.base_model import BaseModel

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

class CatBoostModel(BaseModel):

    def __init__(self, loss=mean_squared_error, cat_col_name: list=None, **kwargs):
        self.model = CatBoostRegressor(silent=True, **kwargs)
        self.params = kwargs
        self.cat_col_name = cat_col_name if cat_col_name is not None else []
        self.loss = loss

    def __encoded_cat_col(self, df: pd.DataFrame, is_test=False) -> pd.DataFrame:
        local_df = df.copy(deep=True)
        if not is_test:
            self.target_encoder.fit(local_df[self.cat_col_name], local_df[BaseModel.target_col_name])
        local_df[self.cat_col_name] = self.target_encoder.transform(local_df[self.cat_col_name])
        return local_df
    
    @BaseModel.check_columns_decorator()
    def fit(self, df: pd.DataFrame):
        df_local = df.copy(deep=True)
        df_local.sort_values(by=[self.date_col_name], inplace=True)
        columns_dtypes = df_local.dtypes
        category_columns_dtypes = columns_dtypes[columns_dtypes == 'category']
        cat = category_columns_dtypes.index.tolist()

        self.get_objective_func(df, 30, self.params)
        catboost_pool = Pool(df_local.drop(columns=[self.target_col_name, self.date_col_name]),
                             df_local[self.target_col_name],
                             cat_features=cat)
        self.model.fit(catboost_pool)
    
    @BaseModel.check_columns_decorator(need_cols=[BaseModel.date_col_name])
    def predict(self, df: pd.DataFrame):
        df_local = df.copy(deep=True)
        df_local.sort_values(by=[self.date_col_name], inplace=True)
        if self.target_col_name in df_local.columns:
            df_local.drop(self.target_col_name, inplace=True)

        columns_dtypes = df_local.dtypes
        category_columns_dtypes = columns_dtypes[columns_dtypes == 'category']
        cat = category_columns_dtypes.index.tolist()

        catboost_pool = Pool(df_local.drop(columns=[self.date_col_name]),
                             cat_features=cat)
        df_local[self.prediction_col_name] = self.model.predict(catboost_pool)
        return df_local
    
    def score(self, y_true, y_pred) -> float:
        return self.loss(y_true, y_pred)
    
    def get_objective_func(self, data, count_days, kwargs, n_trials=20, loss=mean_squared_error):

        def objective(trial):
            nonlocal data
            nonlocal kwargs
            train, test = self.train_test_split(data, count_of_days=count_days)

            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05),
                'depth': trial.suggest_int('depth', 1, 11),
                'random_strength': trial.suggest_float('random_strength', 1e-5, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 20),
                'iterations': trial.suggest_int('iterations', 100, 1000)
            }

            combined_params = params | kwargs
            model = CatBoostRegressor(**combined_params)

            model.fit(train.drop(columns=[self.target_col_name, self.date_col_name]), train[self.target_col_name])
            pred = model.predict(test.drop(columns=[self.target_col_name, self.date_col_name]))
            return loss(pred, test[self.target_col_name])
        
        study = optuna.create_study(direction='minimize')

        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_params = best_params | kwargs
        self.model = CatBoostRegressor(**best_params)
        return best_params

    

