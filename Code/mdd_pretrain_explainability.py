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

## Define Model

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

##################################################################################################################################################################################

# Spectral Explainability Function
def Perturbation_Freq(model,X,Y,N_Reps):
    
    N = np.shape(X)[0]
    Fs = 250 # Sampling Rate
    timestep = 1/Fs # Step Size
    
    # Define Frequency Bins
    bins = [];
    bins.append([0,4]) # delta
    bins.append([4,8]) # theta
    bins.append([8,12]) # alpha
    bins.append([12,25]) # beta
    bins.append([25,50]) # gamma
    bins = np.array(bins)
    
    initial_pred = np.argmax(model.predict(X, batch_size=128),axis=1)
    
    bacc_1 = balanced_accuracy_score(Y,initial_pred)
    
    freq = np.fft.fftfreq(np.shape(X)[1], d=timestep) # 6250 sample frequencies
    
    # Identify Frequency Values Associated with Each Frequency Bin
    bins2 = np.zeros_like(freq) # preallocate array to store marker that identifies bin
    
    for bin_val in range(np.shape(bins)[0]): # for each frequency band
        positive = np.logical_and(freq>bins[bin_val,0]*np.ones_like(freq),freq<bins[bin_val,1]*np.ones_like(freq)) # indices between positive frequencies
        negative = np.logical_and(freq<-1*bins[bin_val,0]*np.ones_like(freq),freq>-1*bins[bin_val,1]*np.ones_like(freq)) # indices between negative frequencies
        vals = positive + negative # all samples within bin (OR the arrays)
        bins2[vals] = bin_val*np.ones((np.sum(vals),)) # assign marker to frequency values in each bin
    
    # Perturbation Explainability
    
    bacc_change = np.zeros((5,N_Reps))
    
    #perform fft for all channels
    fft_vals = np.fft.fft(X_test,axis=1)
    
    for bin_val in range(np.shape(bins)[0]): # iterate over each frequency band
        
        # Duplicate Samples
        fft_vals2 = fft_vals.copy()
        
        for rep in range(N_Reps):    
            np.random.seed(rep)
            perm_idx = np.random.permutation(N)

            # Permute Frequency Values
            fft_perm = fft_vals2[:,np.squeeze(list(bins2 == bin_val*np.ones_like(bins2))),:]
            fft_vals2[:,np.squeeze(list(bins2 == bin_val*np.ones_like(bins2)))] = fft_perm[perm_idx,...]

            # Convert Perturbed Samples Back to Time Domain
            feature_ifft = np.fft.ifft(fft_vals2,axis=1);
            X_2 = feature_ifft

            after_pred = np.argmax(model.predict(X_2, batch_size=128),axis=1)
            bacc_2 = balanced_accuracy_score(Y,after_pred)

            bacc_change[bin_val,rep] = 100*(bacc_2 - bacc_1)/bacc_1
            
            print('Freq ' + str(bin_val) + ' - Rep ' + str(rep))
                        
    return (bacc_change)

###################################################################################################################################

# Spatial Perturbation Function

def Perturbation_Channel(model,X,Y):
    
    N = np.shape(X)[0] 
    N_Timepoints = np.shape(X)[1]
    N_Channels = np.shape(X)[2]
    
    initial_pred = np.argmax(model.predict(X, batch_size=128),axis=1)
    
    bacc_1 = balanced_accuracy_score(Y,initial_pred)
    
    # Perturbation Explainability
    
    bacc_change = np.zeros((19,1))
    
    for channel in np.arange(N_Channels):
            X_2 = X.copy()
            
            # Replace channel with zeros
            X_2[:,:,channel] = np.zeros((N,N_Timepoints))

            after_pred = np.argmax(model.predict(X_2, batch_size=128),axis=1)
            bacc_2 = balanced_accuracy_score(Y,after_pred)

            bacc_change[channel] = 100*(bacc_2 - bacc_1)/bacc_1
            
            print('Channel ' + str(channel))
                        
    return (bacc_change)

######################################################################################################################################################
# Explainability Analysis

# Model A
gss = GroupShuffleSplit(n_splits = 10, train_size = 0.9, random_state = 11) # 1

spectral_importance = []; spatial_importance = [];
fold = 0
for tv_idx, test_idx in gss.split(data, labels, groups):
    
    print(fold)
    
    tf.keras.backend.clear_session()

    X_test = data[test_idx]
    y_test = labels[test_idx]
    
    file_path = "/data/users2/cellis42/Spectral_Explainability/PreTraining/Journal/Models/Models_v2/model_a_seed3_fold"+str(fold)+".hdf5"
 
    N_Reps = 100
    
    # Basic Model Spectral Importance
    K.clear_session()
    model = get_model()
    model.compile(optimizer=keras.optimizers.Adam(0.00075), loss='categorical_crossentropy',
                 metrics = ['acc'])

    model.load_weights(file_path)
    
    spectral_importance.append(Perturbation_Freq(model,X_test,y_test,N_Reps))
    spatial_importance.append(Perturbation_Channel(model,X_test,y_test))

    fold += 1

save_path = "/data/users2/cellis42/Spectral_Explainability/PreTraining/Journal/Models/Models_v2/Importance/model_a_importance_seed3.mat"
savemat(save_path,{"spectral_importance":spectral_importance,"spatial_importance":spatial_importance})

# Models B through D

for md in range(3):
    if md == 0: # Model B
        string = "b"
    elif md == 1: # Model C
        string = "c"
    elif md == 2: # Model D
        string = "d"
    
    save_path = "/data/users2/cellis42/Spectral_Explainability/PreTraining/Journal/Models/Models_v2/model_" + string + "_importance_seed3.mat"
    
    spectral_importance = []; spatial_importance = [];
    for sleep_model_idx in range(10): # iterate over each sleep model
        fold = 0

        gss = GroupShuffleSplit(n_splits = 25, train_size = 0.9, random_state = 3) # 1
        
        spectral_importance_folds = []; spatial_importance_folds = [];
        for tv_idx, test_idx in gss.split(data, labels, groups):
            
            file_path = "/data/users2/cellis42/Spectral_Explainability/PreTraining/Journal/Models/Models_v2/model_" + string + "_sleepmodel_0" + str(sleep_model_idx) + "_fold0"+str(fold)+".hdf5"
            
            tf.keras.backend.clear_session()
            X_test = data[test_idx]
            y_test = labels[test_idx]

            # Model Importance

            K.clear_session()
            model = get_model()
            model.load_weights(file_path)
            model.compile(optimizer=keras.optimizers.Adam(0.00075), loss='categorical_crossentropy',
                         metrics = ['acc'])
            
            spectral_importance_folds.append(Perturbation_Freq(model,X_test,y_test,N_Reps))
            spatial_importance_folds.append(Perturbation_Channel(model,X_test,y_test))
            fold += 1

        spectral_importance.append(spectral_importance_folds)
        spatial_importance.append(spatial_importance_folds)
    savemat(save_path,{"spectral_importance":spectral_importance,"spatial_importance":spatial_importance})
