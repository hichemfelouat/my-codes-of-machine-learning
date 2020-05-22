"""
dataset:
https://finance.yahoo.com/quote/AAPL/history?period1=1585699200&period2=1589932800&interval=1d&filter=history&frequency=1d
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ******************************************************************************
# data: Sequence of observations as a list or 2D NumPy array. Required.
# n_in: Number of lag observations as input (X). Values may be between [1..len(data)] Optional. Defaults to 1.
# n_out: Number of observations as output (y). Values may be between [0..len(data)-1]. Optional. Defaults to 1.
# dropnan: Boolean whether or not to drop rows with NaN values. Optional. Defaults to True
# return: Pandas DataFrame of series framed for supervised learning.

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	  n_vars = 1 if type(data) is list else data.shape[1]
	  df = pd.DataFrame(data)
	  cols, names = list(), list()
	  # input sequence (t-n, ... t-1)
	  for i in range(n_in, 0, -1):
		    cols.append(df.shift(i))
		    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	  # forecast sequence (t, t+1, ... t+n)
	  for i in range(0, n_out):
		    cols.append(df.shift(-i))
		    if i == 0:
			      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		    else:
			      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	  # put it all together
	  agg = pd.concat(cols, axis=1)
	  agg.columns = names
	  # drop rows with NaN values
	  if dropnan:
		    agg.dropna(inplace=True)
	  return agg
# ******************************************************************************
# ******************************************************************************
# ******************************************************************************
# 01/01/1995 to 31/03/2020
dat = pd.read_csv("AAPL.csv",index_col=0)
values = dat.values
print("Original dataset  : \n",dat.head())    
dat.plot(figsize=(12,8),subplots=True)

# We choose a specific feature (features). In this example,
# We are only interested in the opening price of the stock
print("-----------------------------------------------------------------------")
dataset = dat[["Open"]]
# dataset = dat[["Open", "High"]]

print("Our new dataset : \n",dataset.head())
# ensure all data is float
values = dataset.astype("float32")

# normalize features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
# reshape into X=t and Y=t+1
print("-----------------------------------------------------------------------")
i_in  = 30 # past observations
n_out = 1 # future observations
reframed = series_to_supervised(scaled, i_in, n_out)
print("Represent the dataset as a supervised learning problem : \n",reframed.head())

# split into train and test sets
# convert an array of values into a dataset matrix
print("-----------------------------------------------------------------------")
values_spl = reframed.values
train_size = int(len(values_spl) * 0.80)
test_size = len(values_spl) - train_size
train, test = values_spl[0:train_size,:], values_spl[train_size:len(values_spl),:]
print("len train and test : ",len(train), len(test))

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
print("train_X shape : ",train_X.shape," train_y shape : ",train_y.shape)
print("test_X shape  : ",test_X.shape, " test_y shape  : ",test_y.shape)

# reshape input to be 3D [samples, timesteps, features]
# The LSTM network expects the input data (X) to be provided with 
# a specific array structure in the form of: [samples, time steps, features].
# Currently, our data is in the form: [samples, features] 
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print("train_X shape 3D : ",train_X.shape," train_y shape : ",train_y.shape)
print("test_X shape  3D : ",test_X.shape, " test_y shape  : ",test_y.shape)
print("-----------------------------------------------------------------------")
# ******************************************************************************
# create and fit the LSTM network
"""
model = keras.models.Sequential()
model.add(keras.layers.LSTM(32, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]) ))
model.add(keras.layers.LSTM(16))
model.add(keras.layers.Dense(1))
"""

"""
model = keras.models.Sequential()
model.add(keras.layers.GRU(8, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]) ))
model.add(keras.layers.GRU(4))
model.add(keras.layers.Dense(1))
"""


model = keras.models.Sequential()
model.add(keras.layers.GRU(16, return_sequences=True, activation="relu",kernel_initializer="he_normal", 
            recurrent_initializer="he_normal", dropout=0.15, recurrent_dropout=0.15,
						input_shape=(train_X.shape[1], train_X.shape[2]) ))
model.add(keras.layers.GRU(8, activation="relu",kernel_initializer="he_normal", 
            recurrent_initializer="he_normal", dropout=0.15, recurrent_dropout=0.15, implementation=2 ))
model.add(keras.layers.Dense(1, activation="relu",kernel_initializer="glorot_uniform"))

model.summary()
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="mean_squared_error", optimizer =optimizer,metrics=["mse",'mae'])
history = model.fit(train_X, train_y, epochs=50, batch_size=64, validation_split=0.2)

# ******************************************************************************
# plot the learning curves
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************

# Evaluate the model
print("-----------------------------------------------------------------------")
model_evaluate = model.evaluate(test_X, test_y)
print("Loss                   : ",model_evaluate[0])
print("Mean Squared Error     : ",model_evaluate[1])
print("Mean Absolute Error    : ",model_evaluate[2])  


# ******************************************************************************
# make predictions
print("-----------------------------------------------------------------------")
trainPredict = model.predict(train_X)
testPredict  = model.predict(test_X)
print("trainPredict : ",trainPredict.shape)
print("testPredict  : ",testPredict.shape)
print("-----------------------------------------------------------------------")
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY       = scaler.inverse_transform([train_y])

testPredict = scaler.inverse_transform(testPredict)
testY       = scaler.inverse_transform([test_y])


# Finally, we can generate predictions using the model for both the train and test 
# dataset to get a visual indication of the skill of the model. 
# Because of how the dataset was prepared, we must shift the predictions so that 
# they align on the x-axis with the original dataset. Once prepared, the data is 
# plotted, showing the original  dataset in blue, the predictions for the training 
# dataset in green, and the predictions on the unseen test dataset in red.

# shift train predictions for plotting
trainPredictPlot = np.empty_like(values)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[i_in:len(trainPredict)+i_in, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(values)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+i_in:len(values), :] = testPredict

# plot baseline and predictions
plt.figure(figsize=(12,8))
plt.plot(scaler.inverse_transform(values), color="blue", label="Actual Stock Price")
plt.plot(scaler.inverse_transform(trainPredictPlot), color="green", label="Predicted Stock Price")
plt.plot(scaler.inverse_transform(testPredictPlot) , color="red", label="Predicted test Stock Price")
plt.title("Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()

# Show the major grid lines with dark grey lines
plt.grid(b=True, which="major", color="#666666", linestyle="-")
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)

plt.show()

# ******************************************************************************
