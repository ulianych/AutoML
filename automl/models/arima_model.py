import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from pmdarima.arima import AutoARIMA
from sklearn.metrics import mean_squared_error
import copy
from preprocess.fill_skip_dates import FillSkipDates
from models.base_model import BaseModel

class ARIMAModel(BaseModel):

    def __init__(self, loss=mean_squared_error, group_by_columns: list = []):
        self.loss = loss
        self.model = AutoARIMA()
        self.group_by_columns = group_by_columns
        self.models_list = {}

    def get_best_model(self, train: pd.Series):
        parametres = [1, 4, 12]
        best_score = float('inf')
        for mm in parametres:
            model = auto_arima(train, start_p=0, start_q=0,
                               max_p=5, max_q=5,
                               start_P=0, start_Q=0,
                               maxP=5, max_Q=5,
                               seasonal=True,
                               m=mm,
                               trace=True,
                               error_action='warn',
                               suppress_warnings=False,
                               stepwise=False,
                               n_fits=50)
            score = model.bic()
            
            if score < best_score:
                best_model = model
                best_score = score
        return best_model
    
    @BaseModel.check_columns_decorator(need_cols=[BaseModel.target_col_name])
    def fit(self, df: pd.DataFrame):
        df_copy = copy.deepcopy(df)

        unique_pairs = df_copy.groupby(self.group_by_columns) if self.group_by_columns else [('all', df_copy)]
        for pair, data in unique_pairs:

            params = self.get_best_model(data.set_index(data[BaseModel.date_col_name])['target']).get_params()

            model = AutoARIMA(order=(params['order']),
                              m=params['seasonal_order'][3],
                              method=params['method'],
                              with_intercept=params['with_intercept'],
                              maxiter=params['maxiter'],
                              scoring=params['scoring'])
            
            self.models_list[pair] = model.fit(data.set_index(data[BaseModel.date_col_name])['target'])

    def predict(self, df: pd.DataFrame):
        all_prediction_dfs = []

        for pair, model in self.models_list.items():
            predictions = model.predict(len(df[self.date_col_name].unique()))

            predictions_df = pd.DataFrame(predictions, columns=[str(pair) + '_prediction'])

            all_prediction_dfs.append(predictions_df)

        final_predictionns_df = pd.concat(all_prediction_dfs, ignore_index=True)

        return final_predictionns_df
    
    def score(self, y_true, y_pred) -> float:
        return self.loss(y_true, y_pred)