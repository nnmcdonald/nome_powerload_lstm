import math
import numpy
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt

# Keeps track of loss per batch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = ([],[])
        self.i = 0

    def on_batch_end(self, batch, logs={}):
        self.losses[0].append(self.i)
        self.losses[1].append(logs.get('loss'))
        self.i += 1

f = open("nome_city_feeder.csv", "r")
data = f.readlines()
f.close()

data = [data[i].rstrip('\n').split(',') for i in range(1, len(data))]

# Convert the data to float values
for i in range(len(data)):
    for j in range(1, len(data[i])):
        # I assigned each unique wind direction a number from 1 to 17
        # these values will be normalized later
        if j == 5:
            if data[i][5] == 'North':
                data[i][5] = 1.0
            elif data[i][5] == 'South':
                data[i][5] = 2.0
            elif data[i][5] == 'East':
                data[i][5] = 3.0
            elif data[i][5] == 'West':
                data[i][5] = 4.0
            elif data[i][5] == 'WNW':
                data[i][5] = 5.0
            elif data[i][5] == 'SSE':
                data[i][5] = 6.0
            elif data[i][5] == 'ENE':
                data[i][5] = 7.0
            elif data[i][5] == 'NE':
                data[i][5] = 8.0
            elif data[i][5] == 'WSW':
                data[i][5] = 9.0
            elif data[i][5] == 'SE':
                data[i][5] = 10.0
            elif data[i][5] == 'SSW':
                data[i][5] = 11.0
            elif data[i][5] == 'NW':
                data[i][5] = 12.0
            elif data[i][5] == 'ESE':
                data[i][5] = 13.0
            elif data[i][5] == 'SW':
                data[i][5] = 14.0
            elif data[i][5] == 'NNE':
                data[i][5] = 15.0
            elif data[i][5] == 'NNW':
                data[i][5] = 16.0
            elif data[i][5] == '':
                data[i][5] = 17.0
        else:
            data[i][j] = float(data[i][j])

# plots loss per batch
def plot_loss(x, y):
    plt.figure()
    plt.subplot()
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.plot(x, y)

def plot_predictions_vs_expected(predictions, y_expected):
    plt.figure()
    plt.subplot()
    plt.xlabel("Time (1 unit=10 min)")
    plt.ylabel("Amps (scaled to range 1 to 0)")
    plt.plot([i for i in range(len(predictions))],[y_expected[i][0][0] for i in range(len(predictions))], label="Expected Values")
    plt.plot([i for i in range(len(predictions))],[i[0][0] for i in predictions], label="Predictions")
    plt.legend()

def plot_expected_values(exp):
    plt.figure()
    plt.subplot()
    plt.ylim(0,1)
    plt.xlabel("Time (1 unit=10 min)")
    plt.ylabel("Amps (scaled to range 0 to 1)")
    plt.title("Expected Values")
    plt.plot([i for i in range(len(exp))],[exp[i][0][0] for i in range(len(exp))])

def plot_predicted_values(pred):
    plt.figure()
    plt.subplot()
    plt.ylim(0,1)
    plt.xlabel("Time (1 unit=10 min)")
    plt.ylabel("Amps (scaled to range 1 to 0)")
    plt.title("Predicted Values")
    plt.plot([i for i in range(len(pred))],[pred[i][0][0] for i in range(len(pred))], color='orange')

# adjust the input instances so the average amps are over time_period*10 minutes
def average_amps(train_data, validation_data, time_period):
    t = [([train_data[i][0]] + [float(sum(j[1] for j in train_data[(i-time_period+1):(i+1)])/time_period)] + train_data[i][2:]) for i in range(time_period-1, len(train_data))]
    v = [([validation_data[i][0]] + [float(sum(j[1] for j in validation_data[i-time_period+1:i+1])/time_period)] + validation_data[i][2:]) for i in range(time_period-1, len(validation_data))]
    return t, v

# pred_len is the time period to predict (example: each instance is 
# the average over 10 min, so pred_len=1 will predict the next 
# instance amp value given an instance vector, pred_len=4 will predict the
# average over the next 40 minutes given an instance),
# layers is the number of hidden layers to add to the network,
# nodes is the nodes per layer, if drop is a positive value it dropout is added,
# eps is the number of training epochs, x values are the instance vectors, 
# y values are the expected amp values for the next time step,
# if no_amp is true then the amp values aren't included in the input vectors,
# returns the lstm model
def get_lstm_model(train_data, validation_data, layers, nodes, drop, eps, b_size, no_amp):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    start_index = 1
    if no_amp:
        start_index = 2

    x_train_data = train_data[:-1]
    # The expected value, y, for a given x is is the average amps over the next n instances
    # where n=pred_len, if pred_len=1 the expected value is the amp value of the next instance
    y_train_data = [train_data[i][1] for i in range(1, len(train_data))]
    x_validation_data = validation_data[:-1]
    y_validation_data = [validation_data[i][1] for i in range(1, len(validation_data))]

    # scale the data to range 0 to 1 and store as numpy.array
    x_train = numpy.array([[i] for i in scaler.fit_transform([x_train_data[i][start_index:] for i in range(len(x_train_data))])])
    y_train = numpy.array([[i] for i in scaler.fit_transform([[y_train_data[i]] for i in range(len(y_train_data))])])

    x_test = numpy.array([[i] for i in scaler.fit_transform([x_validation_data[i][start_index:] for i in range(len(x_validation_data))])])
    y_test = numpy.array([[i] for i in scaler.fit_transform([[y_validation_data[i]] for i in range(len(y_validation_data))])])

    # Create the model
    lstm_model = Sequential()
    # input layer
    lstm_model.add(LSTM(nodes, input_shape=(1, len(x_train[0][0])), return_sequences=True))
    # hidden layers
    for i in range(layers):
        lstm_model.add(LSTM(nodes, return_sequences=True))
    if drop > 0:
        lstm_model.add(Dropout(drop))

    # output layer
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mse', optimizer='adam')
    # history stores the loss per batch
    history = LossHistory()
    lstm_model.fit(x_train, y_train, batch_size=b_size, epochs=eps, validation_data=(x_test, y_test), callbacks=[history])

    predictions = lstm_model.predict(x_test)

    # create desired plots
    plot_expected_values(y_test)
    plot_predicted_values(predictions)
    plot_predictions_vs_expected(predictions, y_test)
    plot_loss(history.losses[0], history.losses[1])
    plt.show()
    return lstm_model

# split the data, 75% for training, 25% for validation
train_data = data[:int(len(data)*0.75)]
validation_data = data[int(len(data)*0.75):]

# input instances are 10 minutes apart, have 9 weather features and 
# one amp measurement feature which is the average for the 
# time period being predicted

# lstm to predict average amps over next ten minutes
next_ten_min_lstm = get_lstm_model(train_data, validation_data, 1, 100, 0, 2, 64, False)
next_ten_min_lstm_no_amp = get_lstm_model(train_data, validation_data, 1, 120, 0, 1, 64, True)

# lstm to predict average amps over next forty minutes
t, v = average_amps(train_data, validation_data, 4)
next_forty_min_lstm = get_lstm_model(t, v, 2, 120, 0.2, 2, 300, False)
next_forty_min_lstm_no_amp = get_lstm_model(t, v, 2, 100, 0, 2, 400, True)

# lstm to predict average amps over next 12 hours
t, v = average_amps(train_data, validation_data, 72)
next_twelve_hour_lstm = get_lstm_model(t, v, 2, 120, 0.2, 2, 400, False)
next_twelve_hour_lstm_no_amp = get_lstm_model(t, v, 2, 100, 0, 2, 400, True)

# lstm to predict average amps over next 24 hours
t, v = average_amps(train_data, validation_data, 144)
next_24_hour_lstm = get_lstm_model(t, v, 3, 110, 0.2, 3, 500, False)
next_24_hour_lstm_no_amp = get_lstm_model(t, v, 3, 120, 0.2, 3, 400, True)


