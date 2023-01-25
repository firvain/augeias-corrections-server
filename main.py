# This is a sample Python script.
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import requests
import xmltodict as xmltodict
from colorama import Fore, init
from matplotlib.dates import DateFormatter
from numpy import set_printoptions
from pandas import to_datetime

from Modules.Scheduler import my_schedule
from Modules.Utils import get_data, set_seed
from sklearn.preprocessing import MinMaxScaler

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
base_url = os.environ.get("ADDVANTAGE_URL")
my_timezone = pytz.timezone('Europe/Athens')


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

    agg = {'Wind speed 100 Hz': "mean", 'RH': "mean",
           'Air temperature': "mean",
           'Leaf Wetness': "mean", 'Soil conductivity_25cm': "mean",
           'Soil conductivity_15cm': "mean",
           'Soil conductivity_5cm': "mean",
           'Soil temperature_25cm': "mean",
           'Soil temperature_15cm': "mean",
           'Soil temperature_5cm': "mean", 'Soil moisture_25cm': "mean",
           'Soil moisture_15cm': "mean", 'Soil moisture_5cm': "mean",
           'Precipitation': "sum", 'Pyranometer': "mean"
           }
    df3 = get_new_addvantage_data(drop_nan=True, aggreg=agg, past_hours=24)
    df3 = df3[[col2]]
    df3.rename(columns={col2: f'addvantage_{col2}'}, inplace=True)
    df3.index = df3.index.tz_convert('UTC')

    df2 = pd.concat([df2, df3], axis=0)
    df2 = df2[~df2.index.duplicated(keep='first')]
    df2.sort_index(inplace=True)


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

    a = latest_weather_data[col1].values

    latest_weather_data[col1] = latest_weather_data[col1] + model_predictions[0, :latest_weather_data.shape[0]]

    b = latest_weather_data[col1].values
    print(f'{Fore.GREEN}Correction made to latest weather data')
    #
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(rotation=90)
    plt.plot(latest_weather_data.index, a, label='actual')
    plt.plot(latest_weather_data.index, b, label='corrected')
    plt.title(f'{col1} Correction')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return latest_weather_data[col1], start_date, end_date


def get_addvantage_session_id():
    response = requests.get(
        f"{base_url}function=login&user=biokoz&passwd=adupi&mode=t&version=1.2")
    obj = xmltodict.parse(response.content)
    return obj["response"]["result"]["string"]


def logout_addvantage(session_id=""):
    response = requests.get(
        f"{base_url}function=logout&session-id={session_id}&mode=t")
    print(f"addvantage logout status code: {Fore.CYAN}{response.status_code}")


def get_addvantage_data_from_server(session_id, sensor_id, past_hours=12):
    slots = int(past_hours * 60 / 5)
    current_datetime = to_datetime('today')
    current_minus = current_datetime - timedelta(hours=past_hours)
    before_datetime = current_minus.replace(microsecond=0)
    print(f"before_datetime: {before_datetime}")
    response = requests.get(
        f"{base_url}function=getdata&session-id={session_id}&id={sensor_id}&date={before_datetime.strftime('%Y%m%dT%H:%M:%S')}&slots={slots}&cache=y&mode=t")
    json_dict = xmltodict.parse(response.content)

    json_WWTP = {}
    measurements = {}
    diagnostics = {}
    counter = 0
    jsonDict = {}
    titles = ["Wind speed 100 Hz", "RH", "Air temperature", "Leaf Wetness", "Soil conductivity_25cm",
              "Soil conductivity_15cm", "Soil conductivity_5cm", "Soil temperature_25cm", "Soil temperature_15cm",
              "Soil temperature_5cm", "Soil moisture_25cm", "Soil moisture_15cm", "Soil moisture_5cm", "Precipitation",
              "Pyranometer", "Current of Terminal A", "Relative Humidity Internal", "Data Delay", "GSM Signal Strength",
              "Radio Error Rate (Long-Term)", "Radio Error Rate (Short-Term)", "Temperature Internal",
              "Charging Regulator", "Battery Voltage"]

    for k in range(15):

        if "v" in json_dict["response"]["node"][k]:
            if isinstance(json_dict["response"]["node"][k]["v"], list):
                maxN = len(json_dict["response"]["node"][k]["v"])

                json_object = [{} for x in range(maxN)]
                for i in json_dict["response"]["node"][k]["v"]:
                    json_object[counter]['value'] = i['#text']
                    json_object[counter]['time'] = i['@t']
                    counter = counter + 1
                measurements[titles[k]] = json_object
                counter = 0
            else:
                jsonDict['time'] = json_dict["response"]["node"][k]["v"]["@t"]
                jsonDict['value'] = json_dict["response"]["node"][k]["v"]["#text"]
                measurements[titles[k]] = jsonDict
        else:
            measurements[titles[k]] = json_dict["response"]["node"][k]["error"]["@msg"]

    for k in range(15, 24):
        if "v" in json_dict["response"]["node"][k]:
            if isinstance(json_dict["response"]["node"][k]["v"], list):
                maxN = len(json_dict["response"]["node"][k]["v"])
                json_object = [{} for x in range(maxN)]
                for i in json_dict["response"]["node"][k]["v"]:
                    json_object[counter]['value'] = i['#text']
                    json_object[counter]['time'] = i['@t']
                    counter = counter + 1
                diagnostics[titles[k]] = json_object
                counter = 0
            else:
                jsonDict['time'] = json_dict["response"]["node"][k]["v"]["@t"]
                jsonDict['value'] = json_dict["response"]["node"][k]["v"]["#text"]
                diagnostics[titles[k]] = jsonDict
        else:
            diagnostics[titles[k]] = json_dict["response"]["node"][k]["error"]["@msg"]

    json_WWTP["measurements"] = measurements

    json_WWTP["diagnostics"] = diagnostics

    json_data = json.dumps(json_WWTP)

    dicti = json.loads(json_data)

    qa = {'timestamp': []}
    for i in dicti['measurements']:

        qa[i] = []
        for idx, item in enumerate(dicti['measurements'][i]):
            if idx == 0:
                item['time'] = datetime.strptime(item["time"], "%Y%m%dT%H:%M:%S")
                item['time'] = my_timezone.localize(item['time'])
            else:

                item["time"] = dicti['measurements'][i][idx - 1]["time"] + timedelta(seconds=int(item["time"][1:]))

            qa['timestamp'].append(item['time'].isoformat())

            qa[i].append(item['value'])

    qa['timestamp'] = sorted(set(qa['timestamp']))
    for i, k in enumerate(qa):

        if (len(qa[k]) < len(qa['timestamp'])) and (k != "timestamp"):
            qa[k].extend([0.0 for x in range(len(qa['timestamp']) - len(qa[k]))])

    df = pd.DataFrame.from_dict(qa)

    df["timestamp"] = to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df.apply(pd.to_numeric)


def resample_dataset(df, aggreg={}):
    if aggreg:
        return df.resample("1h").agg(aggreg).bfill()
    else:
        return df.resample("1h").mean().bfill()


def get_new_addvantage_data(
        drop_nan=False, aggreg={},
        past_hours=12):
    session_id = get_addvantage_session_id()

    data = get_addvantage_data_from_server(session_id, sensor_id=7608, past_hours=past_hours)
    logout_addvantage(session_id)
    data['Wind speed 100 Hz'] = np.where(
        (data['Wind speed 100 Hz'] < 0.8) & (data['Wind speed 100 Hz'] > 60.0),
        np.nan, data['Wind speed 100 Hz'])
    data['RH'] = np.where(
        (data['RH'] < 0.0) | (data['RH'] > 100.0),
        np.nan, data['RH'])
    data['Air temperature'] = np.where(
        (data['Air temperature'] < -40.0) | (data['Air temperature'] > 60.0),
        np.nan, data['Air temperature'])
    data['Leaf Wetness'] = np.where(
        (data['Leaf Wetness'] < 0.0) | (data['Leaf Wetness'] > 10.0),
        np.nan, data['Leaf Wetness'])
    data['Soil moisture_25cm'] = np.where(
        (data['Soil moisture_25cm'] < 0.0) | (data['Soil moisture_25cm'] > 100.0),
        np.nan, data['Soil moisture_25cm'])
    data['Soil moisture_15cm'] = np.where(
        (data['Soil moisture_15cm'] < 0.0) | (data['Soil moisture_15cm'] > 100.0),
        np.nan, data['Soil moisture_15cm'])
    data['Soil moisture_5cm'] = np.where(
        (data['Soil moisture_5cm'] < 0.0) | (data['Soil moisture_5cm'] > 100.0),
        np.nan, data['Soil moisture_5cm'])
    data['application_group'] = '68ead743e6d6e531352fe86280918678761982bc'
    if drop_nan:
        data.dropna(how='all', inplace=True)
    df_out = resample_dataset(data, aggreg)
    df_out['application_group'] = '68ead743e6d6e531352fe86280918678761982bc'

    return df_out


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
        # evapotranspiration, _, _ = if_you_want_loyalty_buy_a_dog(new_train=False, col1='evapotranspiration', col2='eto',
        #                                                          table='accuweather_direct',
        #                                                          past_steps=72, future_steps=12)

        df = pd.concat([temp, rh, precipitation, wind_speed], axis=1)
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