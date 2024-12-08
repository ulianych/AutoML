import numpy as np
import pandas as pd
from base_preprocess import BasePreprocess
from models.base_model import BaseModel
import holidays
import datetime

import warnings
warnings.filterwarnings('ignore')

class CalenderFilter(BasePreprocess):

    def __init__(self, holiday=False, first_second_working_day=False, salary=False,
                 day_number_m=False, week_number_y = False, week_number_m = False,
                 day_number=False, month=False, special_days=[]):
        self.holiday = holiday
        self.first_second_working_day = first_second_working_day
        self.salary = salary
        self.week_number_y = week_number_y
        self.week_number_m = week_number_m
        self.day_number = day_number
        self.day_number_m = day_number_m
        self.month = month

        self.holidays_by = holidays.BY()
        self.sat_index = 5
        self.sun_index = 6
        self.date = BaseModel.date_col_name
        self.special_days = special_days

        self.salary_days = ['10', '15', '20', '25']
        self.__curr_date = datetime.datetime(2004, 9, 29)

    def add_holiday(self, df: pd.DataFrame):
        new_df = df.copy(deep=True)
        new_df['is_holiday'] = new_df[self.date].apply(lambda x: (str(x) in self.holidays_by) 
                                                       or (x.weekday() in (self.sat_index, self.sun_index)))
        new_df['is_holiday'] = new_df['is_holiday'].astype(int)
        return new_df
    
    def __get_first_work_date(self, line):
        if ((line[self.date] not in self.holidays_by.values())
            and (line[self.date].weekday() not in (self.sat_index, self.sun_index))):
            self.__curr_date = line[self.date]
        return self.__curr_date() if str(line[self.date].day) in self.salary_days else None
    
    def add_salary_day(self, df: pd.DataFrame):
        new_df = df.copy(deep=True)
        self.__curr_date = new_df[self.date].min()

        new_df['first_work_day'] = new_df.apply(lambda x: self.__get_first_work_date(x), axis=1)
        new_df['salary_day'] = new_df[self.date].isin(pd.to_datetime(new_df['first_work_day']))
        new_df['salary_day'] = new_df['salary_day'].astype(int)
        new_df.drop(columns=['first_work_day'], axis=1, inplace=True)
        return new_df
    
    def add_first_second_work_day(self, df: pd.DataFrame):
        new_df = df.copy(deep=True)
        holiday_df = pd.to_datetime(list(self.holidays_by.keys()))
        new_df['month'] = new_df[self.date].dt.to_period('M')
        new_df['first_second_working_day'] = 0
        working_days = new_df[~(new_df[self.date].isin(holiday_df))]

        working_days['weekend'] = working_days[self.date].apply(lambda x: x.weekday() in (self.sat_index, self.sun_index))
        working_days['weekend'] = working_days['weekend'].astype(int)
        working_days = working_days[working_days['weekend'] == 0]
        working_days.drop(columns='weekend', axis=1, inplace=True)

        working_days_unique = working_days.drop_duplicates(self.date)
        first_second = working_days_unique.groupby('month')[self.date].nsmallest(2).reset_index()
        first_second = pd.DataFrame(first_second)
        first_second['first_second_working_day'] = 1
        first_second.drop(columns=['month', 'level_1'], axis=1, inplace=True)
        new_df = pd.merge(new_df, first_second, on=self.date, how='left')

        new_df.drop(columns=['first_second_working_day_x', 'month'], axis=1, inplace=True)
        new_df['first_second_working_day_y'] = new_df['first_second_working_day_y'].apply(lambda x: 0 if pd.isna(x) else 1)
        new_df.rename(columns={'first_second_working_day_y': 'first_second_working_day'}, inplace=True)
        new_df['first_second_working_day'] = new_df['first_second_working_day'].astype(int)
        return new_df
    
    def add_numbers(self, df: pd.DataFrame):
        new_df = df.copy(deep=True)
        if self.week_number_y:
            new_df['week_number_year'] = new_df[self.date].apply(lambda x: x.isocalendar()[1])
        if self.day_number:
            new_df['day_of_week'] = new_df[self.date].apply(lambda x: x.weekday())
        if self.month:
            new_df['month'] = new_df[self.date].apply(lambda x: x.month)
        if self.week_number_m:
            week_number_month = (new_df[self.date].apply(lambda x: 
                                                         (x.isocalender()[1] - datetime.datetime(x.year, x.moth, 1).isocalender()[1])+1))
            new_df['week_number_month'] = week_number_month
        if self.day_number_m:
            new_df['day_number'] = new_df[self.date].apply(lambda x: x.day)
        return new_df
    
    def add_special_day(self, df: pd.DataFrame):
        new_df = df.copy(deep=True)
        new_df['special_days'] = pd.to_datetime(new_df[self.date]).dt.weekday.isin(self.special_days)
        new_df['special_days'] = new_df['special_days'].astype(int)
        return new_df
    
    def fit(self, df: pd.DataFrame):
        pass

    @BaseModel.check_columns_decorator(need_cols=[BaseModel.date_col_name])
    def transform(self, df: pd.DataFrame):
        min_date = df[self.date].min().date()
        max_date = df[self.date].max().date()
        delta = (max_date - min_date).days + 1

        if delta != df[self.date].nunique():
            raise ValueError('Error: missing dates.')
        
        df_transformed = df.copy(deep=True)
        if self.holiday:
            df_transformed = self.add_holiday(df_transformed)
        if self.first_second_working_day:
            df_transformed = self.first_second_working_day(df_transformed)
        if self.salary:
            df_transformed = self.add_salary_day(df_transformed)
        if len(self.special_days):
            df_transformed = self.add_special_day(df_transformed)
        df_transformed = self.add_numbers(df_transformed)
        return df_transformed
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        df = self.transform(df)
        return df
    
