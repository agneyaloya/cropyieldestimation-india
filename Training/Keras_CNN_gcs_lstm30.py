from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
#from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import sklearn
from sklearn import metrics
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.models import load_model
import csv


## Loading the cleaned images downloaded from the MODIS Satellite for 4 counties in 2003. This is to inspect the
## dimensions of the cleaned images.

## We then convert our images to histogram buckets so that each X image has dimensions 32*32*9 (32 buckets, 9 bands, 32 timesteps)
## Loading and inspecting the dimensions of the histogram npz files

image_hist = np.load('./cleaned_rice_data.npz')

X_all = image_hist["output_image"]
Y_all = image_hist["output_yield"]

X_all = X_all.transpose((0,2,1,3))
X_all = X_all.reshape(X_all.shape[0],X_all.shape[1],-1)

s = np.arange(X_all.shape[0])
np.random.seed(seed=5)
np.random.shuffle(s)

X_shuffled = X_all[s]
Y_shuffled = Y_all[s]

#total data set = 1323
#training 1099 i.e. 0:1099
#dev 112 i.e. 1099:1211
#test 107 i.e. 1211:1318 

#intensity bands only
X_train = X_shuffled[0:1099, : , :]
Y_train = Y_shuffled[0:1099].reshape(1099, 1)

X_val = X_shuffled[1099:1211, : , :]
Y_val = Y_shuffled[1099:1211].reshape(112, 1)

X_test = X_shuffled[1211:1318, : , :]
Y_test = Y_shuffled[1211:1318].reshape(107, 1)

# Checking shape of x_train and y_train
print(X_train.shape)
print(Y_train.shape)

print(X_val.shape)
print(Y_val.shape)

print(X_test.shape)
print(Y_test.shape)


def error_metric(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [np.mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return (1 - (squared_error_regr / squared_error_y_mean))


batch_size = 25
epochs = 100
img_rows, channels = 32, 288  # input image dimensions

input_shape = (img_rows, channels)

# no of filters maybe reduced
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1, mode='auto')

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=input_shape))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(1))

#opt_adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
opt_adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=False)

model.compile(loss=keras.losses.mean_squared_error, optimizer=opt_adam, metrics=[error_metric, 'mape'])

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_data=(X_val, Y_val), callbacks=[early_stop])
#history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, Y_val))

'''
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Dropout(0.1))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.1))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.2)))
model.add(Dropout(0.1))
model.add(Dense(1))
'''

f = open('images/run30/output_Adam.txt','w')

## Running on validation data
y_pred_train = model.predict(X_train)
r2error = coefficient_of_determination(Y_train, y_pred_train)
print('R2 error for train set:', r2error[0], file=f)

## train data
score_val = model.evaluate(X_train, Y_train, verbose=1)
print('Loss on training', score_val[0], file=f)
print('RMSE on training', score_val[1], file=f)
print('MAPE on training:', score_val[2], file=f)

## validation data
score_val = model.evaluate(X_val, Y_val, verbose=1)
print('Loss on validation', score_val[0], file=f)
print('RMSE on dev', score_val[1], file=f)
print('MAPE on dev:', score_val[2], file=f)

## test data
#score_test = model.evaluate(X_test, Y_test, verbose=1)
#print('Loss on test', score_test[0], file=f)
#print('RMSE on test:', score_test[1], file=f)
#print('MAPE on test:', score_test[2], file=f)


## Running on validation data

y_pred_val = model.predict(X_val)
y_pred_train = model.predict(X_train)

model.save('images/run30/Keras_CNN_gcs_adam_run30.h5')

xval = list(range(112))

#pp = PdfPages('images/output_images_Adam3.pdf')

plt.figure(0)
plt.plot(xval, Y_val, 'r--',label="Y actual")
plt.plot(xval, y_pred_val, 'b--',label="Y predicted")
plt.legend(loc ="upper left")
plt.xlabel("Observations")
plt.ylabel("Yield")
plt.title("Predicted vs Actual Yield for Validation Set")
#pp.savefig()
plt.savefig('images/run30/YpredvsYactual_Simpler_Adam.png')
#fig1.show()

plt.figure(1)
plt.scatter(Y_val, y_pred_val)
plt.xlabel("True validation vals (Y_val)")
plt.ylabel("Predicted validation vals (Y_pred_val")
plt.title("Predicted vs Actual Yield for Validation Set")
#pp.savefig()
plt.savefig('images/run30/YpredvsYactual_Scatter_Adam.png')

plt.figure(2)
plt.scatter(Y_train, y_pred_train)
plt.xlabel("True training vals (Y_train)")
plt.ylabel("Predicted training vals (Y_pred_train")
plt.title("Predicted vs Actual Yield for Train Set")
#pp.savefig()
plt.savefig('images/run30/YpredvsYactualtrain_Scatter_Adam.png')

plt.figure(3)
plt.plot(history.history['val_mean_absolute_percentage_error'])
plt.xlabel("Epoch")
plt.ylabel("MAPE")
plt.title("MAPE for validation set")
# plt.ylim( 0, 300 )
#pp.savefig()
plt.savefig('images/run30/MAPE_simpler_Adam.png')

plt.figure(4)
plt.plot(history.history['val_error_metric'])
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("RMSE for validation set")
#pp.savefig()
plt.savefig('images/run30/RMSE_simpler_Adam.png')
#plt.ylim(0,2)

plt.figure(5)
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend(loc ="upper left")
plt.title("Loss for Training vs Validation")
plt.savefig('images/run30/Loss_history_trainNval_Adam.png')

#Import CSV
### Output the actual Y, and the Predicted Y for Validation
predictions = np.concatenate((Y_val, y_pred_val), axis = 1)
myFile = open('images/run30/predictions_Adam.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(predictions)
print("Writing complete")

## Running on test data

#y_pred_test = model.predict(X_test)

#print(history.history.keys())