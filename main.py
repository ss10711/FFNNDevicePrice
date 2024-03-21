# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
'''''
brand = x.device_brand.astype('category').cat.codes
os = x.os.astype('category').cat.codes
X = np.column_stack()
screen_size = x['screen_size']
4g = x['4g']
5g = x['5g']
rear_camera_mp
front_camera_mp
internal_memory
ram
battery
weight
release_year
days_used
normalized_new_price
You are given a used electronic devices price with different attributes such as brand, os, screen size, rear-camera, front-camera, normalized used price, etc. 
Please consider normalized used price as your target variable (y). 
Perform a regression analysis task by designing and developing feed-forward neural network.
Your designed feed forward neural network performs regression task and compare performance of linear regression and your neural network(s) performance using MAE, MSE 
and RMSE metric score. Your model can have 2, 3, 4 layers and number of neurons randomly selected from the sample [128, 32, 64, 16]. 
For example, you have 3 layers neural network, you can have 64 neurons in first layer, 32 neurons in second layer, 128 neuron in third layer. 

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
'''''
import math
import random

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
tf.random.set_seed(1)
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
#1 = columns

data = pd.read_csv(r"C:\Users\selen\.conda\cleaned_used_device_data.csv")
x = pd.DataFrame(data.drop(['normalized_used_price'],axis=1))
print(x)
y = pd.DataFrame(data['normalized_used_price'])
print(y)
X_train, X_val, y_train, y_val = train_test_split(x, y)
y_train=np.reshape(y_train, (-1,1))
y_val=np.reshape(y_val, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X_train))
xtrain_scale=scaler_x.transform(X_train)
print(scaler_x.fit(X_val))
xval_scale=scaler_x.transform(X_val)
print(scaler_y.fit(y_train))
ytrain_scale=scaler_y.transform(y_train)
print(scaler_y.fit(y_val))
yval_scale=scaler_y.transform(y_val)


listOfNeurons = [128, 32, 64, 16]
neurons1 = random.choice(listOfNeurons)
listOfNeurons.remove(neurons1)
neurons2 = random.choice(listOfNeurons)
listOfNeurons.remove(neurons2)
neurons3 = random.choice(listOfNeurons)

model = Sequential()
model.add(Dense(neurons1, input_dim=14, kernel_initializer='normal', activation='relu'))
model.add(Dense(neurons2, activation='relu'))
model.add(Dense(neurons3, activation='linear'))
model.add(Dense(1,activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError(),'mse','mae'])
history=model.fit(X_train, y_train, epochs=30, batch_size=150, verbose=1, validation_split=0.2)
predictions = model.predict(X_val)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(mean_squared_error(y_val,predictions))
print(mean_absolute_error(y_val,predictions))
print(math.sqrt(mean_squared_error(y_val,predictions)))
