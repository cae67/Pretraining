# Deep Learning Libraries

from functools import partial
import keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, GlobalMaxPool1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Activation, concatenate, SpatialDropout1D, TimeDistributed, Layer
from keras.utils import to_categorical
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

# Generic ML Libraries
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

# General Libraries
import numpy as np
from scipy.io import loadmat, savemat
from scipy.fft import fft, fftfreq, ifft
import h5py
import os

# Figure Libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap


## IMPORT DATA

datadir = "/home/users/cellis42/Spectral_Explainability"
filename = "segmented_sc_data.mat"

with h5py.File(os.path.join(datadir,filename), 'r') as mat_file:
    X = np.expand_dims(np.float32(mat_file['X'].value.T),axis=2); # data
    Y = np.float32(mat_file['Y'].value.T); # labels
    S = np.float32(mat_file['subject'].value.T); # subject number

print(np.shape(X))
print(np.shape(Y))
print(np.shape(S))

X = X[np.squeeze(S < 39*np.ones_like(S)),...] # Get samples from first half of subjects
Y = Y[np.squeeze(S < 39*np.ones_like(S)),...]
S = S[np.squeeze(S < 39*np.ones_like(S)),...]
print(np.shape(X))
print(np.shape(Y))
print(np.shape(S))

# Remove Marked Samples
X = X[np.squeeze(Y)!=5*np.ones_like(np.squeeze(Y)),...]
S = S[np.squeeze(Y)!=5*np.ones_like(np.squeeze(Y)),...]
Y = Y[np.squeeze(Y)!=5*np.ones_like(np.squeeze(Y)),...]
print(np.shape(X))
print(np.shape(Y))
print(np.shape(S))

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
    
def get_model(dropout=0.5):
    
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

    return model

model = get_model()
print(model.summary)
model = []

# Function for Fitting New Model and Outputting Model and Training History
def evaluate_model(X_train, X_val, Y_train, Y_val,checkpoint):
    model = get_model()
    early = EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, verbose=2) #adaptive learning 
    callbacks_list = [checkpoint, early, redonplat] 
    
    #%% Create Weights for Model Classes
    values, counts = np.unique(Y_train, return_counts=True)

    weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), np.squeeze(Y_train))
    class_weights = dict(zip(values, weights))
    
    X_train = np.repeat(X_train,repeats=19,axis=2)
    X_train = X_train + np.random.normal(loc=0, scale=0.7, size=X_train.shape)
    
    print('Created Training Set')
    
    X_val = np.repeat(X_val,repeats=19,axis=2)
    X_val = X_val + np.random.normal(loc=0, scale=0.7, size=X_val.shape)
    
    print('Created Validation Set')
    
    model = get_model()
    
    model.compile(optimizer=keras.optimizers.Adam(0.00075), loss='categorical_crossentropy',
                     metrics = ['accuracy'])
    
    history = model.fit(X_train, to_categorical(np.array(Y_train)), epochs=200, batch_size=128, shuffle=True, validation_data=(X_val, to_categorical(np.array(Y_val))), verbose = 0,callbacks=callbacks_list,class_weight=class_weights)
       
    return model, history


# Run Classifier for 10 Folds
n_folds = 10
Y_pred = []; Y_test_all = []; Y_pred_val = []; Y_val_all = [];
val_loss = []; train_loss = [];
val_acc = []; train_acc = [];
Sample_Idx = np.expand_dims(np.arange(np.shape(Y)[0]),axis=1) 

count = 0
# split data into Train/Val and Test Groups
cv = GroupShuffleSplit(n_splits=10,test_size=0.1,train_size=0.9,random_state=0)
for train_val_idx, test_idx in cv.split(X,Y,S):
    X_train_val = X[train_val_idx,...]
    Y_train_val = Y[train_val_idx,...]
    S_train_val = S[train_val_idx,...]
    X_test = X[test_idx,...]
    Y_test = Y[test_idx,...]
    S_test = S[test_idx,...]
    Sample_Idx_Test = Sample_Idx[test_idx,...]
    
    # Split Train/Val Data into Training and Validation Groups
    cv2 = GroupShuffleSplit(n_splits=1,test_size=0.10,train_size=0.90,random_state=0)
    for train_idx, val_idx in cv2.split(X_train_val,Y_train_val,S_train_val):
        X_train = X_train_val[train_idx,...]
        Y_train = Y_train_val[train_idx,...]
        S_train = S_train_val[train_idx,...]
        X_val = X_train_val[val_idx,...]
        Y_val = Y_train_val[val_idx,...]
        S_val = S_train_val[val_idx,...]
    X_train_val = []; Y_train_val = []; S_train_val = []
    
    # Define Model Checkpoints
    file_path = "/home/users/cellis42/Spectral_Explainability/PreTraining/Models/sleep_model_Fold"+str(count)+".hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # Evaluate model
    model, history = evaluate_model(X_train, X_val, Y_train, Y_val,checkpoint)
    
    print('Train Acc')
    print(np.max(history.history['acc']))
    print('Val Acc')
    print(np.max(history.history['val_acc']))
    
    val_loss.append(history.history['val_loss']); train_loss.append(history.history['loss'])
    val_acc.append(history.history['val_acc']); train_acc.append(history.history['acc']);

    # Load Weights of Best Model for Fold
    model.load_weights(file_path)

    Y_pred.append(np.argmax(model.predict(np.repeat(X_test,repeats=19,axis=2)),axis=1))
    Y_test_all.append(Y_test)
    
    Y_pred_val.append(np.argmax(model.predict(np.repeat(X_val,repeats=19,axis=2)),axis=1))
    Y_val_all.append(Y_val)

    # Save Test Data for Fold
    # dc = {'X':X_test, 'Y':Y_test, 'subject': S_test, 'Sample_Idx':Sample_Idx_Test}
    # savemat("/home/users/cellis42/Spectral_Explainability/PreTraining/test_data" + str(count) + ".mat",dc)
    
    print(count)
    count += 1
  

# Output In Depth Validation Results
n_folds = 10
conf_mat = np.zeros((5,5))
precision = np.zeros((5,n_folds))
recall = np.zeros((5,n_folds))
f1 = np.zeros((5,n_folds))
f1_ind = []

for i in range(n_folds):
    conf_mat += confusion_matrix(Y_val_all[i],Y_pred_val[i])
    metrics = np.array(sklearn.metrics.precision_recall_fscore_support(Y_val_all[i], Y_pred_val[i], beta=1.0, average=None))
    precision[:,i] = np.array(metrics)[0,:]
    recall[:,i] = np.array(metrics)[1,:]
    f1[:,i] = np.array(metrics)[2,:]
    f1_ind.append(sklearn.metrics.f1_score(Y_val_all[i], Y_pred_val[i],average = 'weighted')) # Compute Weighted F1 Score

print('Validation Results')
print(np.int64(conf_mat))
print('Precision Mean')
print(np.mean(precision,axis=1))
print('Precision SD')
print(np.std(precision,axis=1))
print('Recall Mean')
print(np.mean(recall,axis=1))
print('Recall SD')
print(np.std(recall,axis=1))
print('F1 Mean')
print(np.mean(f1,axis=1))
print('F1 SD')
print(np.std(f1,axis=1))

# Output In Depth Test Results
n_folds = 10
conf_mat = np.zeros((5,5))
precision = np.zeros((5,n_folds))
recall = np.zeros((5,n_folds))
f1 = np.zeros((5,n_folds))
f1_ind = []

for i in range(n_folds):
    conf_mat += confusion_matrix(Y_test_all[i],Y_pred[i])
    metrics = np.array(sklearn.metrics.precision_recall_fscore_support(Y_test_all[i], Y_pred[i], beta=1.0, average=None))
    precision[:,i] = np.array(metrics)[0,:]
    recall[:,i] = np.array(metrics)[1,:]
    f1[:,i] = np.array(metrics)[2,:]
    f1_ind.append(sklearn.metrics.f1_score(Y_test_all[i], Y_pred[i],average = 'weighted')) # Compute Weighted F1 Score

print('Test Results')
print(np.int64(conf_mat))
print('Precision Mean')
print(np.mean(precision,axis=1))
print('Precision SD')
print(np.std(precision,axis=1))
print('Recall Mean')
print(np.mean(recall,axis=1))
print('Recall SD')
print(np.std(recall,axis=1))
print('F1 Mean')
print(np.mean(f1,axis=1))
print('F1 SD')
print(np.std(f1,axis=1))
