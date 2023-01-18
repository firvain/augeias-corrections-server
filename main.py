# This is a sample Python script.
import sys
from pathlib import Path

import numpy as np
from colorama import Fore, init
from numpy import set_printoptions, sqrt
from sklearn.metrics import mean_squared_error

from Modules.Utils import get_data, multivariate_data, set_seed
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from matplotlib import rcParams, pyplot as plt
import tensorflow as tf

init(autoreset=True)
set_seed(42)
set_printoptions(suppress=True)
rcParams['figure.figsize'] = (20, 10)


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def get_data_from_db(table_name, column, name):
    data = get_data(table_name)
    if data is not None:
        print(f'{Fore.GREEN}Data loaded')
        data = data[[column]]
        print(data)
        data.rename(columns={column: f'{name}_{column}'}, inplace=True)
        return data
    else:
        print(f'{Fore.RED}No data found')
        return None


def prepare_data(data, past_steps, future_steps, step):
    x_train, y_train = multivariate_data(data.values, data['diff'].values, 0,
                                         TRAIN_SPLIT, past_steps,
                                         future_steps, step)
    x_val, y_val = multivariate_data(data.values, data['diff'].values,
                                     TRAIN_SPLIT, None, past_steps,
                                     future_steps, step)
    return x_train, y_train, x_val, y_val


def munch_data(data1, data2):
    data = data1.join(data2)
    data.dropna(inplace=True)
    data['diff'] = data[f'{data1.columns[0]}'] - data[f'{data2.columns[0]}']
    return data


def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(future_target))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def train_model(model, x_train, y_train, x_val, y_val, use_es=True):
    if use_es:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    else:
        es = None
    return model.fit(x_train, y_train, epochs=20,
                     batch_size=1, validation_data=(x_val, y_val),
                     verbose=1, shuffle=False, callbacks=[es])


def plot_history(model_history):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def predict(model, x_val, y_val):
    predictions = []
    for i in range(0, x_val.shape[0]):
        yhat = model.predict(x_val[i].reshape(1, x_val.shape[1], x_val.shape[2]))
        yhat = scaler.inverse_transform(yhat.reshape(-1, 1))
        y_val[i] = scaler.inverse_transform(y_val_multi[i].reshape(1, -1))
        predictions.append(yhat)

    predictions = np.array(predictions)
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])
    print(predictions.shape)
    print(predictions)
    return predictions


def plot_diff_predictions(predictions, y_val, column, save=False, show=False):
    hours = [i for i in range(0, future_target)]

    for hour in hours:
        rmse = sqrt(mean_squared_error(y_val[:, hour], predictions[:, hour]))
        plt.plot(y_val[:, hour], label='actual')
        plt.plot(predictions[:, hour], label='predicted')
        plt.title(
            f'Target: {col1}_diff - Dataset:{table} - RMSE: {rmse:.2f} - Future hour: {hour + 1} - RMSE: {rmse:.2f}')
        plt.xlabel('Epoch')
        plt.ylabel(f'{column} Difference')
        plt.legend()
        if save:
            plt.savefig(f'{path}/predictions/hour_{hour + 1}.png')
        if show:
            plt.show()


def plot_corrections(original, predictions, column, table_name, save=False, show=False):
    hours = [i for i in range(0, future_target)]
    for hour in hours:
        o = original.values.reshape(original.shape[0])
        p = predictions[:, hour]
        correction = o + p
        rmse = sqrt(mean_squared_error(o, correction))
        plt.plot(o, label=f'Original {column}', color='blue', linestyle='-')
        plt.plot(correction, label=f'Corrected {column}', color='green', linestyle='--')
        plt.title(f'Corrected: {column}_diff - Dataset:{table_name} - Future hour: {hour + 1} - RMSE: {rmse:.2f}')
        plt.xlabel('Epoch')
        plt.ylabel('Temperature Corrections')
        plt.legend()
        if save:
            plt.savefig(f'{path}/corrections/hour_{hour + 1}.png')
        if show:
            plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    table = "accuweather_direct"
    col1 = 'rh'
    df1 = get_data_from_db(table, col1, 'opendataset')
    col2 = 'RH'
    df2 = get_data_from_db('addvantage', col2, 'addvantage')
    if df1 is not None and df2 is not None:
        df = munch_data(df1, df2)
        df.to_csv(f'Data/{table}_addvantage_diff.csv')
    else:
        print(f'{Fore.RED}No data to process')
        sys.exit(1)

    # prepare data
    past_history = 72
    future_target = 24
    STEP = 1
    TRAIN_SPLIT = int(len(df) * 0.9)
    x_train_multi, y_train_multi, x_val_multi, y_val_multi = prepare_data(df, past_history, future_target, STEP)
    print(f'{Fore.GREEN}Data prepared')
    print(f'{Fore.GREEN}x_train shape: {x_train_multi.shape}')
    print(f'{Fore.GREEN}y_train shape: {y_train_multi.shape}')
    print(f'{Fore.GREEN}x_val shape: {x_val_multi.shape}')
    print(f'{Fore.GREEN}y_val shape: {y_val_multi.shape}')
    print('Single window of past history : {}'.format(x_train_multi[-1].shape))
    print('Single window of future target : {}'.format(y_train_multi[-1].shape))

    # scale data
    scaler = MinMaxScaler()
    x_train_multi = scaler.fit_transform(x_train_multi.reshape(-1, 1)).reshape(x_train_multi.shape)
    x_val_multi = scaler.transform(x_val_multi.reshape(-1, 1)).reshape(x_val_multi.shape)
    y_train_multi = scaler.fit_transform(y_train_multi.reshape(-1, 1)).reshape(y_train_multi.shape)
    y_val_multi = scaler.transform(y_val_multi.reshape(-1, 1)).reshape(y_val_multi.shape)
    print(f'{Fore.CYAN}Data scaled')
    print(f'{Fore.CYAN}x_train shape: {x_train_multi.shape}')
    print(f'{Fore.CYAN}y_train shape: {y_train_multi.shape}')
    print(f'{Fore.CYAN}x_val shape: {x_val_multi.shape}')
    print(f'{Fore.CYAN}y_val shape: {y_val_multi.shape}')



    # set Paths
    path = f"Plots/{table}/{col1}/{future_target}_future_hours"
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path + '/predictions').mkdir(parents=True, exist_ok=True)
    Path(path + '/corrections').mkdir(parents=True, exist_ok=True)

    NEW_MODEL = False
    # create  new model
    if NEW_MODEL:
        my_model = create_model(input_shape=x_train_multi.shape[-2:])
        history = train_model(my_model, x_train_multi, y_train_multi, x_val_multi, y_val_multi)
        plot_history(history)
        my_model.save(f'Models/{col1}_{future_target}_addvantage_diff.h5')
    else:
        my_model = tf.keras.models.load_model(f'Models/{col1}_{future_target}_addvantage_diff.h5')

    # make predictions
    my_model = tf.keras.models.load_model(f'Models/{col1}_{future_target}_addvantage_diff.h5')
    model_predictions = predict(my_model, x_val_multi, y_val_multi)
    print(model_predictions.shape)
    print(model_predictions)

    # plot predictions
    plot_diff_predictions(model_predictions, y_val_multi, col1, save=True, show=False)
    # plot corrections
    org = df.iloc[-y_val_multi.shape[0]:][f'opendataset_{col1}']
    plot_corrections(org, model_predictions, col1, table, save=True, show=False)

