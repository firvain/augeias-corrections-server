# This is a sample Python script.
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from colorama import Fore, init
from matplotlib.dates import DateFormatter
from numpy import set_printoptions, sqrt
from sklearn.metrics import mean_squared_error

from Modules.Scheduler import my_schedule
from Modules.Utils import get_data, set_seed
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from matplotlib import rcParams, pyplot as plt

from dotenv import load_dotenv
from sqlalchemy import create_engine
import tensorflow as tf

init(autoreset=True)
set_seed(42)
set_printoptions(suppress=True)
rcParams['figure.figsize'] = (20, 10)

load_dotenv('.env')
POSTGRESQL_URL = os.environ.get("POSTGRESQL_URL")
engine = create_engine(POSTGRESQL_URL)


def get_data_from_db(table_name, column, name, limit=None):
    data = get_data(table_name, limit)
    if data is not None:
        print(f'{Fore.GREEN}{table_name} Data loaded')
        data = data[[column]]
        data.rename(columns={column: f'{name}_{column}'}, inplace=True)
        return data
    else:
        print(f'{Fore.RED}No data found')
        return None


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []
    data_index = []
    labels_index = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset.values) - target_size

    for i in range(start_index, end_index + 1):
        indices = range(i - history_size, i, step)

        data.append(dataset.values[indices])
        data_index.append(dataset.index[indices])

        if single_step:
            labels.append(target.values[i + target_size])
            labels_index.append(target.index[i + target_size])
        else:
            labels.append(target.values[i:i + target_size])
            labels_index.append(target.index[i:i + target_size])

    return np.array(data), np.array(labels), np.array(data_index), np.array(labels_index)


def prepare_data(data, past_steps, future_steps, step, train_split):
    x_train, y_train, x_train_index, y_train_index = multivariate_data(data, data['diff'], 0,
                                                                       train_split, past_steps,
                                                                       future_steps, step)
    x_val, y_val, x_val_index, y_val_index = multivariate_data(data, data['diff'],
                                                               train_split, None, past_steps,
                                                               future_steps, step)

    return x_train, y_train, x_train_index, y_train_index, x_val, y_val, x_val_index, y_val_index


def munch_data(data1, data2):
    data = data1.join(data2)
    data.dropna(inplace=True)
    data['diff'] = data[f'{data1.columns[0]}'] - data[f'{data2.columns[0]}']
    return data


def predict(model, x_val, scaler):
    if len(x_val.shape) == 2:
        x_val = np.expand_dims(x_val, axis=0)

    predictions = []
    for i in range(len(x_val)):
        yhat = model.predict(x_val[i].reshape(1, x_val.shape[1], x_val.shape[2]), verbose=0)
        # invert scaling for forecast
        yhat = scaler.inverse_transform(yhat.reshape(-1, 1))

        predictions.append(yhat)

    predictions = np.array(predictions)
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])

    return predictions


def if_you_want_loyalty_buy_a_dog(new_train=False, col1=None, col2=None, table=None, past_steps=72, future_steps=12, ):
    Path('Models').mkdir(parents=True, exist_ok=True)
    Path('Data').mkdir(parents=True, exist_ok=True)
    Path('Plots').mkdir(parents=True, exist_ok=True)
    # table = "accuweather_direct"
    # col1 = col1
    df1 = get_data_from_db(table, col1, 'opendataset')
    # col2 = 'Air temperature'
    df2 = get_data_from_db('addvantage', col2, 'addvantage')
    if df1 is not None and df2 is not None:
        df = munch_data(df1, df2)
        df.to_csv(f'Data/{table}_addvantage_{col1}_diff.csv')
    else:
        print(f'{Fore.RED}No data to process')
        sys.exit(1)

    # prepare data
    past_history = past_steps
    future_target = future_steps
    STEP = 1
    TRAIN_SPLIT = int(len(df) * 0.9)

    x_train_multi, y_train_multi, x_train__multi_index, y_train_multi_index, x_val_multi, y_val_multi, x_val_multi_index, y_val_multi_index = prepare_data(
        df, past_history, future_target, STEP, TRAIN_SPLIT)



    # scale data
    scaler = MinMaxScaler()
    x_train_multi = scaler.fit_transform(x_train_multi.reshape(-1, 1)).reshape(x_train_multi.shape)
    # x_val_multi = scaler.transform(x_val_multi.reshape(-1, 1)).reshape(x_val_multi.shape)
    # y_train_multi = scaler.fit_transform(y_train_multi.reshape(-1, 1)).reshape(y_train_multi.shape)
    # y_val_multi = scaler.transform(y_val_multi.reshape(-1, 1)).reshape(y_val_multi.shape)
    print(f'{Fore.CYAN}Data scaled')
    # print(f'{Fore.CYAN}x_train shape: {x_train_multi.shape}')
    print(f'{Fore.CYAN}y_train shape: {y_train_multi.shape}')

    # print(f'{Fore.CYAN}x_val shape: {x_val_multi.shape}')
    # print(f'{Fore.CYAN}y_val shape: {y_val_multi.shape}')

    # set Paths
    path = f"Plots/{table}/{col1}/{future_target}_future_hours"
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path + '/predictions').mkdir(parents=True, exist_ok=True)
    Path(path + '/corrections').mkdir(parents=True, exist_ok=True)

    # make predictions
    my_model = tf.keras.models.load_model(f'Models/{col1}_{future_target}_addvantage_diff.h5')

    last_window = df.iloc[-past_steps:, :]

    last_window = last_window.values.reshape(1, last_window.shape[0], last_window.shape[1])
    print(f'{Fore.GREEN}Last window shape: {last_window.shape}')

    inp = last_window.reshape(-1, 1)
    print(f'{Fore.GREEN}inp shape: {inp.shape}')

    inp = scaler.transform(inp)
    inp = inp.reshape(1, past_steps, last_window.shape[2])
    print(f'{Fore.YELLOW}Input shape: {inp.shape}')
    y_train_multi = scaler.fit_transform(y_train_multi.reshape(-1, 1)).reshape(y_train_multi.shape)
    model_predictions = predict(my_model, inp, scaler)
    print(f'{Fore.GREEN}Predictions made')
    print(f'{Fore.GREEN}model_predictions shape: {model_predictions.shape}')

    start_date = df.iloc[-past_steps:, :].index[-1] + timedelta(hours=1)
    end_date = start_date + timedelta(hours=future_target)

    latest_weather_data = get_data(table=table, start_date=start_date, limit=future_target,
                                   end_date=end_date)
    latest_weather_data.sort_index(inplace=True)


    # a = latest_weather_data[col1].values
    latest_weather_data[col1] = latest_weather_data[col1] + model_predictions[0, :]
    # b = latest_weather_data[col1].values
    print(f'{Fore.GREEN}Correction made to latest weather data')
    #
    # ax = plt.gca()
    # ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
    # plt.xticks(rotation=90)
    # plt.plot(latest_weather_data.index, a, label='actual')
    # plt.plot(latest_weather_data.index, b, label='corrected')
    # plt.title(f'{col1} Correction')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    return latest_weather_data[col1], start_date, end_date


def make_corrections():
    try:
        temp, start, end = if_you_want_loyalty_buy_a_dog(new_train=False, col1='temp', col2='Air temperature',
                                                         table='accuweather_direct',
                                                         past_steps=72, future_steps=12)
        rh, _, _ = if_you_want_loyalty_buy_a_dog(new_train=False, col1='rh', col2='RH', table='accuweather_direct',
                                                 past_steps=72, future_steps=12)

        precipitation, _, _ = if_you_want_loyalty_buy_a_dog(new_train=False, col1='precipitation', col2='Precipitation',
                                                            table='accuweather_direct',
                                                            past_steps=72, future_steps=12)
        wind_speed, _, _ = if_you_want_loyalty_buy_a_dog(new_train=False, col1='wind_speed', col2='Wind speed 100 Hz',
                                                         table='accuweather_direct',
                                                         past_steps=72, future_steps=12)
        evapotranspiration, _, _ = if_you_want_loyalty_buy_a_dog(new_train=False, col1='evapotranspiration', col2='eto',
                                                                 table='accuweather_direct',
                                                                 past_steps=72, future_steps=12)

        df = pd.concat([temp, rh, precipitation, wind_speed, evapotranspiration], axis=1)
        Path('Data/corrections').mkdir(parents=True, exist_ok=True)
        df.to_csv(
            f'Data/corrections/corrections_{start.strftime("%Y_%m_%d %H_%M_%S")}_{end.strftime("%Y_%m_%d %H_%M_%S")}.csv')
        df.to_sql('accuweather_corrected', con=engine, if_exists='append', index=True)
    except Exception as e:
        print(e)
        print(f'{Fore.RED}Something went wrong')
        return


if __name__ == '__main__':
    # my_schedule(make_corrections)
    make_corrections()
