import pandas as pd
import numpy as np

from models.base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

class RandomForest(BaseModel):

    def __init__(self, object_columns=[], loss=mean_squared_error, drop_col_name: list=None, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
        self.loss = loss
        self.object_columns = [*object_columns]
        self.drop_col_name = [*drop_col_name] if drop_col_name is not None else []
        self.codes = {}
        self.params = kwargs


    def target_encoder(self, df: pd.DataFrame, column_name, flag):
        if (column_name not in self.codes) and flag:
            raise Exception('No column ' + column_name)
        elif column_name not in self.codes:
            target_means = df.groupby(column_name)[self.target_col_name].mean()
            data_list = target_means.to_dict()
            self.codes[column_name] = data_list

        encoded_df = df.copy()
        target_values = self.codes[column_name]

        if not (set(encoded_df[column_name].unique()) <= set(self.codes[column_name].keys())):
            raise Exception('Imbalance of values in ' + column_name)
        encoded_df[str(column_name + '_encoded')] = encoded_df[column_name].map(target_values)
        encoded_df.drop(columns=column_name, axis=1, inplace=True)
        return encoded_df
    
    def encoder(self, df: pd.DataFrame, flag):
        encoded_df = df.copy()
        for column in encoded_df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                if column == self.date_col_name:
                    continue
                else:
                    encoded_df = self.target_encoder(encoded_df, column, flag)
    
    def transform(self, df: pd.DataFrame, flag):
        df = self.encoder(df, flag)
        return df
    
    @BaseModel.check_columns_decorator()
    def fit(self, df: pd.DataFrame):
        df = self.transform(df, False)
        self.get_objective_func(df, 30, self.params)
        self.model.fit(df.drop(columns=[self.target_col_name, self.date_col_name], axis=1), df[self.target_col_name])

    @BaseModel.check_columns_decorator(need_cols=[BaseModel.date_col_name])
    def predict(self, df: pd.DataFrame):
        raw_df = df.copy(deep=True)
        df = self.transform(df, True)
        test_df = df[:]
        if self.target_col_name in test_df.columns:
            test_df = test_df.drop(self.target_col_name, axis=1)
        pred = self.model.predict(test_df.drop(columns=[self.date_col_name]))
        raw_df[BaseModel.prediction_col_name] = pred
        return raw_df
    
    def score(self, y_true, y_pred) -> float:
        return self.loss(y_true, y_pred)
    
    def get_objective_func(self, data, count_days, kwargs, n_trials=20, loss=mean_squared_error):

        def objective(trial):
            nonlocal data
            nonlocal kwargs
            train, test = self.train_test_split(data, count_of_days=count_days)

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                'max_depth': trial.suggest_int('max_depth', 1, 32),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }

            combined_params = params | kwargs
            model = RandomForestRegressor(**combined_params)

            model.fit(train.drop(columns=[self.target_col_name, self.date_col_name]), train[self.target_col_name])
            pred = model.predict(test.drop(columns=[self.target_col_name, self.date_col_name]))
            return loss(pred, test[self.target_col_name])
        
        study = optuna.create_study(direction='minimize')

        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_params = best_params | kwargs
        self.model = RandomForestRegressor(**best_params)
        return best_params

