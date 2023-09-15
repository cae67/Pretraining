## 4 Models
# MDD Model A - Baseline MDD Model
# MDD Model B - Pretrained MDD Model with Tuned Dense Layers Only
# MDD Model C - Pretrained MDD Model With Tuning of Whole Model B 
# MDD Model D - Pretrained MDD Model Whole Model Tuning

# Deep Learning Libraries

from functools import partial
import keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, GlobalMaxPool1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Activation, concatenate, SpatialDropout1D, TimeDistributed, Layer
from keras.utils import to_categorical
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.model_selection import GroupShuffleSplit
from functools import partial
from keras.callbacks import *
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score

import sklearn
from sklearn.metrics import confusion_matrix

# General Libraries
from scipy.io import loadmat, savemat
from scipy.fft import fft, fftfreq, ifft
import h5py
import os

# import innvestigate as inn
# import innvestigate.utils

# tf.random.set_random_seed(42)

filepath = ['/home/users/cellis42/Spectral_Explainability/PreTraining/segmented_hc1_data_like_sleep.npy',
            '/home/users/cellis42/Spectral_Explainability/PreTraining/segmented_hc2_data_like_sleep.npy',
            '/home/users/cellis42/Spectral_Explainability/PreTraining/segmented_mdd1_data_like_sleep.npy',
            '/home/users/cellis42/Spectral_Explainability/PreTraining/segmented_mdd2_data_like_sleep.npy']

for i in np.arange(4):

    f = np.load(filepath[i],allow_pickle=True).item()
    
    if i == 0:
        data = f['data']
        labels = f['label']
        groups = f['subject']
    else:
        data = np.concatenate((data,f['data']),axis=0)
        labels = np.concatenate((labels,f['label']),axis=0)
        groups = np.concatenate((groups,f['subject']),axis=0)
        channels = f['channels']
                
channels2 = []
for i in range(19):
    channels2.append(channels[i].strip('EEG ').strip('-L'))

channels = channels2
channels2 = []

data = np.swapaxes(data,1,2)

## ChannelDropout Code
class ChannelDropout(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
    
    def call(self, inputs, training = None):
        return tf.nn.dropout(inputs, 
                             rate=self.rate, 
                            noise_shape = [1,1,inputs.shape[2]])

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {
            "rate": self.rate,
            "noise_shape": self.noise_shape,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Define Sleep Model
def get_model_sleep(dropout=0.5):
    
    n_timesteps = 3000
    n_features = 19

    convLayer = partial(keras.layers.convolutional.Conv1D,activation='elu',kernel_initializer='he_normal',padding='valid',
                        kernel_constraint=keras.constraints.max_norm(max_value = 1))
    
    model = keras.models.Sequential()
    model.add(ChannelDropout(rate=0.25))
    model.add(convLayer(5, kernel_size=10, strides=1, input_shape=(n_timesteps, n_features), data_format='channels_last'))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.BatchNormalization())

    model.add(convLayer(10, kernel_size=10, strides=1))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.BatchNormalization())
    
    model.add(convLayer(10, kernel_size=10, strides=1))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.BatchNormalization())
    
    model.add(convLayer(15, kernel_size=5, strides=1))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.AlphaDropout(rate= dropout))
    
    model.add(keras.layers.Dense(64, activation='elu', kernel_initializer='he_normal', kernel_constraint=keras.constraints.max_norm(max_value = 1),name = "dense_l1"))
    model.add(keras.layers.AlphaDropout(rate= dropout))
    model.add(keras.layers.Dense(32, activation='elu', kernel_initializer='he_normal', kernel_constraint=keras.constraints.max_norm(max_value = 1),name="dense_l2"))
    model.add(keras.layers.AlphaDropout(rate= dropout))
    model.add(keras.layers.Dense(5, activation='softmax', kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.max_norm(max_value = 1),name="dense_output"))
    # model.compile(optimizer=keras.optimizers.Adam(0.00075), loss='categorical_crossentropy',
    #                  metrics = ['accuracy'])
    return model

## Define MDD Model

def get_model(dropout=0.5):
    
    n_timesteps = 3000
    n_features = 19

    convLayer = partial(keras.layers.convolutional.Conv1D,activation='elu',kernel_initializer='he_normal',padding='valid',
                        kernel_constraint=keras.constraints.max_norm(max_value = 1))
    
    model = keras.models.Sequential()
    model.add(convLayer(5, kernel_size=10, strides=1, input_shape=(n_timesteps, n_features), data_format='channels_last'))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.BatchNormalization())

    model.add(convLayer(10, kernel_size=10, strides=1))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.BatchNormalization())
    
    model.add(convLayer(10, kernel_size=10, strides=1))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.BatchNormalization())
    
    model.add(convLayer(15, kernel_size=5, strides=1))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.AlphaDropout(rate= dropout))
    
    model.add(keras.layers.Dense(64, activation='elu', kernel_initializer='he_normal', kernel_constraint=keras.constraints.max_norm(max_value = 1),name = "dense_l1"))
    model.add(keras.layers.AlphaDropout(rate= dropout))
    model.add(keras.layers.Dense(32, activation='elu', kernel_initializer='he_normal', kernel_constraint=keras.constraints.max_norm(max_value = 1),name="dense_l2"))
    model.add(keras.layers.AlphaDropout(rate= dropout))
    model.add(keras.layers.Dense(2, activation='softmax', kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.max_norm(max_value = 1),name="dense_output"))

    return model

batch_norm_layer_idx = [2, 5, 8, 11]

##################################################################
# Model A - Baseline Model

tf.random.set_random_seed(41)

testing_metrics = []; validation_metrics = [];

n_timesteps = 3000
n_features = 19

i = 0

val_loss = []; train_loss = [];
val_acc = []; train_acc = [];

gss = GroupShuffleSplit(n_splits = 25, train_size = 0.9, random_state = 3)

for tv_idx, test_idx in gss.split(data, labels, groups):
    
    print(i)

    X_train_val = data[tv_idx]
    y_train_val = labels[tv_idx]

    X_test = data[test_idx]
    y_test = labels[test_idx]

    group = groups[tv_idx]
    gss_train_val = GroupShuffleSplit(n_splits = 1, train_size = 0.83, random_state = 3) #train_size = 0.89, random_state = 11
    for train_idx, val_idx in gss_train_val.split(X_train_val, y_train_val, group):
        
        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        
        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]
        
        # Data Augmentation
        np.random.seed(41)
        X_train = np.vstack((X_train, X_train + np.random.normal(loc=0, scale=0.7, size=X_train.shape)))
        y_train = np.hstack((y_train, y_train))
        
        # K.clear_session()
        model = get_model()
        model.compile(optimizer=keras.optimizers.Adam(0.00075), loss='categorical_crossentropy',
                     metrics = ['acc'])

        file_path = "/data/users2/cellis42/Spectral_Explainability/PreTraining/Journal/Models/Models_v2/model_a_seed3_fold"+str(i)+".hdf5"
        
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc",min_delta=0,patience=10)
        checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

        # Create Weights for Model Classes
        values, counts = np.unique(y_train, return_counts=True)

        weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = np.squeeze(y_train))
        class_weights = dict(zip(values, weights))
        
        history = model.fit(X_train, keras.utils.to_categorical(y_train), epochs= 35, batch_size = 128, validation_data=(X_val, keras.utils.to_categorical(y_val)), 
                            shuffle=True, verbose = 0, callbacks=[checkpoint,early_stopping],class_weight=class_weights)
        
        val_loss.append(history.history['val_loss']); train_loss.append(history.history['loss'])
        val_acc.append(history.history['val_acc']); train_acc.append(history.history['acc']);
        
        model.load_weights(file_path)

        preds = np.argmax(model.predict(X_test, batch_size=128),axis=1)

        testing_metrics.append([accuracy_score(y_test, preds),recall_score(y_test, preds, pos_label=1),recall_score(y_test, preds, pos_label=0),balanced_accuracy_score(y_test, preds)])
        
        preds_val = np.argmax(model.predict(X_val, batch_size=128),axis=1)

        validation_metrics.append([accuracy_score(y_val, preds_val),recall_score(y_val, preds_val, pos_label=1),recall_score(y_val, preds_val, pos_label=0),balanced_accuracy_score(y_val, preds_val)])
    
    i += 1
    
results_filename = "/home/users/cellis42/Spectral_Explainability/PreTraining/Journal/model_a_25_seed3.mat"
savemat(results_filename,{"validation_metrics":validation_metrics,"testing_metrics":testing_metrics})
  
# ##################################################################

print('Model B: Train with Frozen Feature Extraction')

tf.random.set_random_seed(41)

n_timesteps = 3000
n_features = 19

testing_metrics_all = []; validation_metrics_all = [];

val_loss = []; train_loss = [];
val_acc = []; train_acc = [];

gss = GroupShuffleSplit(n_splits = 25, train_size = 0.9, random_state = 3) # 2

for sleep_model_idx in range(10): # iterate over each sleep model
    testing_metrics = []; validation_metrics = [];
    i = 0

    for tv_idx, test_idx in gss.split(data, labels, groups):

        print(i)
        tf.keras.backend.clear_session()

        X_train_val = data[tv_idx]
        y_train_val = labels[tv_idx]

        X_test = data[test_idx]
        y_test = labels[test_idx]

        group = groups[tv_idx]
        gss_train_val = GroupShuffleSplit(n_splits = 1, train_size = 0.83, random_state = 3)
        for train_idx, val_idx in gss_train_val.split(X_train_val, y_train_val, group):

            X_train = X_train_val[train_idx]
            y_train = y_train_val[train_idx]

            X_val = X_train_val[val_idx]
            y_val = y_train_val[val_idx]
            
            # Data Augmentation
            np.random.seed(41)
            X_train = np.vstack((X_train, X_train + np.random.normal(loc=0, scale=0.7, size=X_train.shape)))
            y_train = np.hstack((y_train, y_train))

            K.clear_session()
            model = get_model()
            model.compile(optimizer=keras.optimizers.Adam(0.00075), loss='categorical_crossentropy',
                         metrics = ['acc'])

            model_sleep = get_model_sleep()
            model_sleep.build((None,3000,19))
            file_path_sleep = "/home/users/cellis42/Spectral_Explainability/PreTraining/Models/sleep_model_Fold"+str(sleep_model_idx)+".hdf5"
            model_sleep.load_weights(file_path_sleep)

            # Transfer weights and Freeze Starting Layers
            for layer in range(14):
                model.layers[layer].set_weights(model_sleep.layers[layer+1].get_weights())
                if layer not in batch_norm_layer_idx:
                    model.layers[layer].trainable = False

            file_path = "/data/users2/cellis42/Spectral_Explainability/PreTraining/Journal/Models/Models_v2/model_b_seed3_sleepmodel_0" + str(sleep_model_idx) + "_fold0"+str(i)+".hdf5"

            early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc",min_delta=0,patience=10)
            checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max', save_weights_only=True)

            # Create Weights for Model Classes
            values, counts = np.unique(y_train, return_counts=True)

            weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = np.squeeze(y_train))
            class_weights = dict(zip(values, weights))

            history = model.fit(X_train, keras.utils.to_categorical(y_train), epochs= 35, batch_size = 128, validation_data=(X_val, keras.utils.to_categorical(y_val)), 
                                shuffle=True, verbose = 0, callbacks=[checkpoint,early_stopping],class_weight=class_weights);

            val_loss.append(history.history['val_loss']); train_loss.append(history.history['loss'])
            val_acc.append(history.history['val_acc']); train_acc.append(history.history['acc']);

            model.load_weights(file_path)

            preds = np.argmax(model.predict(X_test, batch_size=128),axis=1)

            testing_metrics.append([accuracy_score(y_test, preds),recall_score(y_test, preds, pos_label=1),recall_score(y_test, preds, pos_label=0),balanced_accuracy_score(y_test, preds)])

            preds_val = np.argmax(model.predict(X_val, batch_size=128),axis=1)

            validation_metrics.append([accuracy_score(y_val, preds_val),recall_score(y_val, preds_val, pos_label=1),recall_score(y_val, preds_val, pos_label=0),balanced_accuracy_score(y_val, preds_val)])
        
        i += 1
        print('Sleep Model ' + str(sleep_model_idx) + ' - Fold ' + str(i))
    validation_metrics_all.append(validation_metrics);
    testing_metrics_all.append(testing_metrics);
    
results_filename = "/home/users/cellis42/Spectral_Explainability/PreTraining/Journal/model_b_25_seed3.mat"
savemat(results_filename,{"validation_metrics_all":validation_metrics_all,"testing_metrics_all":testing_metrics_all})

# print("Validation Set Metrics")
# validation_metrics = np.array(validation_metrics_all)
# print(validation_metrics)
# print(pd.DataFrame(data=[validation_metrics.mean(axis=0), validation_metrics.std(axis=0)], index=['mean','std'], columns=['acc','sens','spec','bacc']))

print("Test Set Metrics")
# print(testing_metrics)
for i in range(10):
    testing_metrics = np.array(testing_metrics_all)[i,...]
    print(pd.DataFrame(data=[testing_metrics.mean(axis=0), testing_metrics.std(axis=0)], index=['mean','std'], columns=['acc','sens','spec','bacc']))

    
##################################################################
print('Model C: Unfreeze Feature Extraction and Continue Training')

tf.random.set_random_seed(41)

testing_metrics_all2 = []; validation_metrics_all2 = [];

n_timesteps = 3000
n_features = 19


val_loss = []; train_loss = [];
val_acc = []; train_acc = [];

for sleep_model_idx in range(10): # iterate over each sleep model
    testing_metrics = []; validation_metrics = [];
    i = 0

    gss = GroupShuffleSplit(n_splits = 25, train_size = 0.9, random_state = 3) # 1

    for tv_idx, test_idx in gss.split(data, labels, groups):

        tf.keras.backend.clear_session()

        X_train_val = data[tv_idx]
        y_train_val = labels[tv_idx]

        X_test = data[test_idx]
        y_test = labels[test_idx]

        group = groups[tv_idx]
        gss_train_val = GroupShuffleSplit(n_splits = 1, train_size = 0.83, random_state = 3)
        for train_idx, val_idx in gss_train_val.split(X_train_val, y_train_val, group):

            X_train = X_train_val[train_idx]
            y_train = y_train_val[train_idx]

            X_val = X_train_val[val_idx]
            y_val = y_train_val[val_idx]
            
            # Data Augmentation
            np.random.seed(41)
            X_train = np.vstack((X_train, X_train + np.random.normal(loc=0, scale=0.7, size=X_train.shape)))
            y_train = np.hstack((y_train, y_train))

            K.clear_session()
            model = get_model()

            file_path = "/data/users2/cellis42/Spectral_Explainability/PreTraining/Journal/Models/Models_v2/model_b_seed3_sleepmodel_0" + str(sleep_model_idx) + "_fold0"+str(i)+".hdf5"
            model.load_weights(file_path)

            # Unfreeze Starting Layers
            for layer in range(14):
                if layer in batch_norm_layer_idx:
                    model.layers[layer].trainable = True
            
            model.compile(optimizer=keras.optimizers.Adam(0.00075), loss='categorical_crossentropy',
                         metrics = ['acc'])

            file_path = "/data/users2/cellis42/Spectral_Explainability/PreTraining/Journal/Models/Models_v2/model_c_seed3_sleepmodel_0" + str(sleep_model_idx) + "_fold0"+str(i)+".hdf5"

            early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc",min_delta=0,patience=10)
            checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max', save_weights_only=True)

            # Create Weights for Model Classes
            values, counts = np.unique(y_train, return_counts=True)

            weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = np.squeeze(y_train))
            class_weights = dict(zip(values, weights))

            history = model.fit(X_train, keras.utils.to_categorical(y_train), epochs= 35, batch_size = 128, validation_data=(X_val, keras.utils.to_categorical(y_val)), 
                                shuffle=True, verbose = 0, callbacks=[checkpoint,early_stopping],class_weight=class_weights);

            val_loss.append(history.history['val_loss']); train_loss.append(history.history['loss'])
            val_acc.append(history.history['val_acc']); train_acc.append(history.history['acc']);

            model.load_weights(file_path)

            preds = np.argmax(model.predict(X_test, batch_size=128),axis=1)

            testing_metrics.append([accuracy_score(y_test, preds),recall_score(y_test, preds, pos_label=1),recall_score(y_test, preds, pos_label=0),balanced_accuracy_score(y_test, preds)])

            preds_val = np.argmax(model.predict(X_val, batch_size=128),axis=1)

            validation_metrics.append([accuracy_score(y_val, preds_val),recall_score(y_val, preds_val, pos_label=1),recall_score(y_val, preds_val, pos_label=0),balanced_accuracy_score(y_val, preds_val)])

        i += 1
        print('Sleep Model ' + str(sleep_model_idx) + ' - Fold ' + str(i))
    validation_metrics_all2.append(validation_metrics);
    testing_metrics_all2.append(testing_metrics);
    
results_filename = "/home/users/cellis42/Spectral_Explainability/PreTraining/Journal/model_c_25_seed3.mat"
savemat(results_filename,{"validation_metrics_all":validation_metrics_all2,"testing_metrics_all":testing_metrics_all2})

# print("Validation Set Metrics")
# validation_metrics = np.array(validation_metrics_all)
# print(validation_metrics)
# print(pd.DataFrame(data=[validation_metrics.mean(axis=0), validation_metrics.std(axis=0)], index=['mean','std'], columns=['acc','sens','spec','bacc']))

print("Test Set Metrics")
# print(testing_metrics)
for i in range(10):
    testing_metrics = np.array(testing_metrics_all2)[i,...]
    print(pd.DataFrame(data=[testing_metrics.mean(axis=0), testing_metrics.std(axis=0)], index=['mean','std'], columns=['acc','sens','spec','bacc']))
    
##################################################################
# Model D (i.e., don't freeze any layers and just train)

tf.random.set_random_seed(41) # best is seed 42, v7

n_timesteps = 3000
n_features = 19

testing_metrics_all = []; validation_metrics_all = [];

val_loss = []; train_loss = [];
val_acc = []; train_acc = [];

gss = GroupShuffleSplit(n_splits = 25, train_size = 0.9, random_state = 3) # 2

for sleep_model_idx in range(10): # iterate over each sleep model
    testing_metrics = []; validation_metrics = [];
    i = 0

    for tv_idx, test_idx in gss.split(data, labels, groups):

        print(i)
        tf.keras.backend.clear_session()

        X_train_val = data[tv_idx]
        y_train_val = labels[tv_idx]

        X_test = data[test_idx]
        y_test = labels[test_idx]

        group = groups[tv_idx]
        gss_train_val = GroupShuffleSplit(n_splits = 1, train_size = 0.83, random_state = 3)
        for train_idx, val_idx in gss_train_val.split(X_train_val, y_train_val, group):

            X_train = X_train_val[train_idx]
            y_train = y_train_val[train_idx]

            X_val = X_train_val[val_idx]
            y_val = y_train_val[val_idx]
            
            # Data Augmentation
            np.random.seed(41)
            X_train = np.vstack((X_train, X_train + np.random.normal(loc=0, scale=0.7, size=X_train.shape)))
            y_train = np.hstack((y_train, y_train))

            K.clear_session()
            model = get_model()

            model_sleep = get_model_sleep()
            model_sleep.build((None,3000,19))
            file_path_sleep = "/home/users/cellis42/Spectral_Explainability/PreTraining/Models/sleep_model_Fold"+str(sleep_model_idx)+".hdf5"
            model_sleep.load_weights(file_path_sleep)

            # Transfer weights
            for layer in range(14):
                model.layers[layer].set_weights(model_sleep.layers[layer+1].get_weights())
                
            model.compile(optimizer=keras.optimizers.Adam(0.00075), loss='categorical_crossentropy',
                         metrics = ['acc'])
            
            file_path = "/data/users2/cellis42/Spectral_Explainability/PreTraining/Journal/Models/Models_v2/model_d_seed3_sleepmodel_0" + str(sleep_model_idx) + "_fold0"+str(i)+".hdf5"

            early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc",min_delta=0,patience=10)
            checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max', save_weights_only=True)

            # Create Weights for Model Classes
            values, counts = np.unique(y_train, return_counts=True)

            weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = np.squeeze(y_train))
            class_weights = dict(zip(values, weights))

            history = model.fit(X_train, keras.utils.to_categorical(y_train), epochs= 35, batch_size = 128, validation_data=(X_val, keras.utils.to_categorical(y_val)), 
                                shuffle=True, verbose = 0, callbacks=[checkpoint,early_stopping],class_weight=class_weights);

            val_loss.append(history.history['val_loss']); train_loss.append(history.history['loss'])
            val_acc.append(history.history['val_acc']); train_acc.append(history.history['acc']);

            model.load_weights(file_path)

            preds = np.argmax(model.predict(X_test, batch_size=128),axis=1)

            testing_metrics.append([accuracy_score(y_test, preds),recall_score(y_test, preds, pos_label=1),recall_score(y_test, preds, pos_label=0),balanced_accuracy_score(y_test, preds)])

            preds_val = np.argmax(model.predict(X_val, batch_size=128),axis=1)

            validation_metrics.append([accuracy_score(y_val, preds_val),recall_score(y_val, preds_val, pos_label=1),recall_score(y_val, preds_val, pos_label=0),balanced_accuracy_score(y_val, preds_val)])
        
        i += 1
        # print('Sleep Model ' + str(sleep_model_idx) + ' - Fold ' + str(i))
    validation_metrics_all.append(validation_metrics);
    testing_metrics_all.append(testing_metrics);
    
results_filename = "/home/users/cellis42/Spectral_Explainability/PreTraining/Journal/model_d_25_seed3.mat"
savemat(results_filename,{"validation_metrics_all":validation_metrics_all,"testing_metrics_all":testing_metrics_all})

# print("Validation Set Metrics")
# validation_metrics = np.array(validation_metrics_all)
# print(validation_metrics)
# print(pd.DataFrame(data=[validation_metrics.mean(axis=0), validation_metrics.std(axis=0)], index=['mean','std'], columns=['acc','sens','spec','bacc']))

print("Test Set Metrics")
# print(testing_metrics)
for i in range(10):
    testing_metrics = np.array(testing_metrics_all)[i,...]
    print(pd.DataFrame(data=[testing_metrics.mean(axis=0), testing_metrics.std(axis=0)], index=['mean','std'], columns=['acc','sens','spec','bacc']))
