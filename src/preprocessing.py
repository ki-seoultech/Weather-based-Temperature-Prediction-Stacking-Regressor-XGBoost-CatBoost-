# ============================================================
# Script: preprocessing.py
# Project: Weather-based Temperature Prediction
# Description:
#   - Loads and cleans weather data
#   - Handles missing values, interpolation
#   - Performs advanced feature engineering
#   - Outputs preprocessed data for modeling
# Author: Hyunbin Ki
# ============================================================


# 1. 라이브러리 임포트
import pandas as pd
import numpy as np
from xgboost import DMatrix, train as xgb_train
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
!pip install catboost
!pip install optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# 2. 데이터 불러오기
train = pd.read_csv("/content/train_dataset.csv")
test = pd.read_csv("/content/test_dataset.csv")
station_info = pd.read_csv("/content/station_info.csv")

# 3. 결측치 처리
train.replace(-9999, np.nan, inplace=True)
test.replace(-9999, np.nan, inplace=True)

# 4. 시간별 피처 결측치 처리
def preprocess_hourly_features(df):
    zero_fill_prefixes = ['sunshine_duration_','snow_depth_','precipitation_','min_cloud_height_']
    interpolate_prefixes = ['surface_temp_', 'humidity_', 'dew_point_', 'vapor_pressure_',
                            'visibility_', 'local_pressure_', 'sea_level_pressure_',
                            'wind_speed_', 'wind_direction_', 'cloud_cover_']



    for prefix in zero_fill_prefixes:
        cols = [col for col in df.columns if col.startswith(prefix)]
        df[cols] = df[cols].fillna(0)

    for prefix in interpolate_prefixes:
        cols = [col for col in df.columns if col.startswith(prefix)]
        df[cols] = df[cols].interpolate(axis=1, limit_direction='both')


    return df

train = preprocess_hourly_features(train)
test = preprocess_hourly_features(test)

# 5. 날짜 처리
train['month'] = pd.to_datetime('2024-' + train['date'], errors='coerce').dt.month
test['month'] = pd.to_datetime('2024-' + test['date'], errors='coerce').dt.month

def add_time_features(df):
    date_parsed = pd.to_datetime('2024-' + df['date'], errors='coerce')
    df['dayofyear'] = date_parsed.dt.dayofyear

    df['weekofyear'] = date_parsed.dt.isocalendar().week.astype(int)
    df['weekday'] = date_parsed.dt.weekday  # 0=월요일
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    return df

train = add_time_features(train)
test = add_time_features(test)

# weekday만 One-Hot (범주형 순서 무의미한 변수)
time_cats = ['weekday']
train = pd.get_dummies(train, columns=time_cats)
test = pd.get_dummies(test, columns=time_cats)

# 컬럼 정렬 맞춤 (train/test 동일하게)
train, test = train.align(test, join='left', axis=1, fill_value=0)

# 6. station_info 병합
station_info.rename(columns={'지점': 'station', '지점명': 'station_name'}, inplace=True)

station_info = station_info.drop_duplicates(['station', 'station_name'])

station_info['시작일'] = pd.to_datetime(station_info['시작일'], errors='coerce')
station_info['종료일'] = pd.to_datetime(station_info['종료일'], errors='coerce')
station_info['운영일수'] = (station_info['종료일'] - station_info['시작일']).dt.days
station_info.drop(columns=['시작일', '종료일'], inplace=True)

train = train.merge(station_info, on=['station', 'station_name'], how='left')
test = test.merge(station_info, on=['station', 'station_name'], how='left')

# 7. One-Hot 인코딩
train = pd.get_dummies(train, columns=['station_name'])
test = pd.get_dummies(test, columns=['station_name'])
train, test = train.align(test, join='left', axis=1, fill_value=0)

#날짜 주기성(계절)
def add_time_features(df):
    date_parsed = pd.to_datetime('2024-' + df['date'], errors='coerce')
    df['dayofyear'] = date_parsed.dt.dayofyear

    df['weekofyear'] = date_parsed.dt.isocalendar().week.astype(int)
    df['weekday'] = date_parsed.dt.weekday  # 0=월요일
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    return df

train = add_time_features(train)
test = add_time_features(test)

# vapor feature 추가
def vapor_pressure_features(df):

    vapor_cols = [f'vapor_pressure_{h}' for h in range(24) if f'vapor_pressure_{h}' in df.columns]
    df['vapor_mean_0_23'] = df[vapor_cols].mean(axis=1)


    #야간 평균 기온 × 밤 평균 vapor
    df['temp_18_23_avg'] = df[[f'surface_temp_{h}' for h in range(18, 24)]].mean(axis=1)
    df['vapor_18_23_avg'] = df[[f'vapor_pressure_{h}' for h in range(18, 24)]].mean(axis=1)
    df['temp_vapor_night_interact'] = df['temp_18_23_avg'] * df['vapor_18_23_avg']

    # 18~23시 평균 vapor_pressure
    vapor_avg = df[[f'vapor_pressure_{h}' for h in range(18, 24) if f'vapor_pressure_{h}' in df.columns]].mean(axis=1)
    df['vapor_18_23_avg'] = vapor_avg

    # 제외할 prefix
    exclude_prefix = 'surface_temp'

    # 대상 prefix들 (surface_temp 제외)
    prefixes = [
        'dew_point', 'humidity',
        'local_pressure', 'sea_level_pressure',
        'wind_speed', 'wind_direction',
        'cloud_cover', 'visibility'
    ]

    for prefix in prefixes:
        try:
            cols = [f"{prefix}_{h}" for h in range(18, 24) if f"{prefix}_{h}" in df.columns]
            if not cols:
                continue  # 존재하지 않는 경우 skip

            df[f'{prefix}_18_23_avg'] = df[cols].mean(axis=1)
            df[f'{prefix}_vapor_night_interact'] = df[f'{prefix}_18_23_avg'] * df['vapor_18_23_avg']
        except Exception as e:
            print(f"{prefix} 처리 중 오류 발생: {e}")
    # 하루 최고/최저 vapor

    df['vapor_min'] = df[vapor_cols].min(axis=1)

    # 오후 → 밤 vapor 변화
    df['vapor_diff_23_12'] = df['vapor_pressure_23'] - df['vapor_pressure_12']


    # 밤 vapor 변화
    df['vapor_night_change'] = df['vapor_pressure_23'] - df['vapor_pressure_0']



    # 대표 시각의 곱 (물리적 상호작용 추정)-
    df['temp23_x_vapor23'] = df['surface_temp_23'] * df['vapor_pressure_23']


    return df


train = vapor_pressure_features(train)
test = vapor_pressure_features(test)




# 8. 기본 파생 피처 생성
def add_hourly_custom_features(df):

    df['temp_night_change'] = df['surface_temp_23'] - df['surface_temp_0']
    df['dew_point_shift'] = df['dew_point_18'] - df['dew_point_6']
    df['dew_point_change_night'] = df['dew_point_23'] - df['dew_point_18']
    df['humidity_change_night'] = df['humidity_23'] - df['humidity_18']
    df['temp_evening_slope'] = df['surface_temp_23'] - df['surface_temp_19']
    df['dew_temp_diff_23'] = df['surface_temp_23'] - df['dew_point_23']
    df['humidity_mean_20_23'] = df[[f'humidity_{h}' for h in range(20, 24)]].mean(axis=1)
    df['surface_temp_std_19_23'] = df[[f'surface_temp_{h}' for h in range(19, 24)]].std(axis=1)
    df['humidity_min_20_23'] = df[[f'humidity_{h}' for h in range(20, 24)]].min(axis=1)
    df['temp_x_humidity_19'] = df['surface_temp_19'] * df['humidity_23']
    df['dew_ratio_23'] = df['dew_point_23'] / (df['surface_temp_19'] + 1e-6)
    df['humidity_log_23'] = np.log1p(df['humidity_23'])
    df['dew_x_humidity_23'] = df['dew_point_23'] * df['humidity_23']
    df['dew_temp_gap_sq_19'] = (df['surface_temp_19'] - df['dew_point_23']) ** 2

    if '노장해발고도(m)' in df.columns:
        df['altitude_adj_temp_19'] = df['surface_temp_19'] + (df['노장해발고도(m)'] / 100 * 0.65)
    return df

train = add_hourly_custom_features(train)
test = add_hourly_custom_features(test)

# 9. 고급 station_info 기반 파생 피처 생성
sensor_cols = [
    '기온계(관측장비지상높이(m))',
    '풍속계(관측장비지상높이(m))',
    '기압계(관측장비지상높이(m))',
    '강우계(관측장비지상높이(m))'
]

def add_advanced_station_features(df):
    df['surface_temp_adjust_ratio'] = df['노장해발고도(m)'] / (df['기온계(관측장비지상높이(m))'] + 1e-3)
    df['wind_exposure_index'] = df['풍속계(관측장비지상높이(m))'] / (df['노장해발고도(m)'] + 1e-3)
    df['sensor_height_std'] = df[sensor_cols].std(axis=1)
    df['night_temp_variation_adjusted'] = df['temp_night_change'] * (df['노장해발고도(m)'] / 100)
    df['dew_x_temp_adjust_ratio'] = df['dew_point_23'] * df['surface_temp_adjust_ratio']

    return df

train = add_advanced_station_features(train)
test = add_advanced_station_features(test)

def add_lat_lon_combination(df):
    if '위도' not in df.columns or '경도' not in df.columns:
        return df

    df['lat_x_lon'] = df['위도'] * df['경도']
    df['lat_div_lon'] = df['위도'] / (df['경도'] + 1e-6)
    df['lat_plus_lon'] = df['위도'] + df['경도']
    df['lat_minus_lon'] = df['위도'] - df['경도']


    return df

train = add_lat_lon_combination(train)
test = add_lat_lon_combination(test)

# polynomial feature
def add_polynomial_features(df):

    # 1. 위도 × 이슬점
    if 'lat' in df.columns and 'dew_point_23' in df.columns:
        df['lat_x_dew23'] = df['lat'] * df['dew_point_23']

    # 2. 위도 × 습도
    if 'lat' in df.columns and 'humidity_23' in df.columns:
        df['lat_x_humidity23'] = df['lat'] * df['humidity_23']

    # 3. 기온 × 습도
    if 'surface_temp_19' in df.columns and 'humidity_23' in df.columns:
        df['temp19_x_humidity23'] = df['surface_temp_19'] * df['humidity_23']

    # 4. 고도 × 야간기온변화
    if '노장해발고도(m)' in df.columns and 'temp_night_change' in df.columns:
        df['altitude_x_temp0'] = df['노장해발고도(m)'] * df['temp_night_change']

    # 5. 이슬점 ÷ 습도 (건조지수)
    if 'dew_point_23' in df.columns and 'humidity_23' in df.columns:
        df['dew_div_humidity'] = df['dew_point_23'] / (df['humidity_23'] + 1e-6)



    return df

train = add_polynomial_features(train)
test = add_polynomial_features(test)

train = train.copy()
test = test.copy()

