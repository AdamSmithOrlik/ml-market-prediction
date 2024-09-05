"""
Script Name: ML_Models.py
Author: Adam Smith-Orlik
Email: asorlik@yorku.ca
Date Created: November 2023
Progress: Development

Description:
Machine Learning Models class used to unify the structure and use of different machine learning models with 
disparate syntax. Script also includes standalone modules used for analysis of ML specific models. The goal
is to call the Models class from a back tester to use it for stock market predictions. 

Usage:
To create an instance of the Models class run "from ML_model import Models"
To load a specific model use "my_model = Models.<input_model>(**hyperparameters)"
An example of the dictionary "hyperparameters" is included as a variable of the 
script. Call "from ML_models import default_hyperparameters. To use the analysis
function call them individually or run "from ML_models import *".
"""
# TENSORFLOW MODELS
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# XGBOOST MODELS
import xgboost as xgb

# SKLEARN FUNCTIONS
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score

# TRACKING RESULTS
import wandb
API_KEY = "6dc78c7577f9146a1f008e276104f2f3afc9fda1"
from wandb.xgboost import WandbCallback as XGBoostWandbCallback
from wandb.keras import WandbCallback as KerasWandbCallback

# MODEL ANALYSIS
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# default dictionary 
default_hyperparameters = {
    'learning_rate': 0.01,
    'epochs': 1000,
    'batch_size': 32,
    'lstm_units': 50, 
    'dense_units': 64,
    'dropout_rate': 0.2,
    'L2_regularize': False,
    'L1_regularize': False,
    'test_split': 0.1,
    'validation_split': 0.4,
    'optimizer': 'Adam',
    'wandb': False, 
    'save': False,
    'name': 'test',
    'run': 'run_01',
    'entity': 'asorliklab',
    'dataset': 'MSFT', 
    'model_architecture': 'Gradient Boosting',
    'objective':'binary:logistic',
    'input_shape':(1, 12)
}

class Model:
    """
    Models class organizes within the same namespace all of the machine learning models that are built to test stock 
    market predictions. This class also standardizes the use of the models that have differing syntax and use cases. 
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, val in self.kwargs.items():
            setattr(self, key, val)

class Binary_XGBoost(Model):
    """
    This model uses XGBClassifier to predict if a security will go up or down tomorrow depending on the data today. 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create model
        self.model = xgb.XGBClassifier(booster='gbtree' ,n_estimators=self.n_estimators, eta=self.learning_rate, 
                                        objective='binary:logistic', eval_metric=['logloss', 'error'])
    
    def train(self, X, Y, Xv, Yv):
        
        if self.wandb: run = wandb_setup(**self.kwargs)

        # wandb compatibility
        self.callback = None if not self.wandb else [XGBoostWandbCallback(log_model=True)]

        self.model.fit(X, Y, eval_set=[(X, Y),(Xv, Yv)],
                    verbose=100,
                    callbacks=self.callback)
        
        if self.wandb: run.finish()

        return
        
    def predict(self, Xt, Yt, threshold=0.5):
        probs = self.model.predict_proba(Xt)
        preds = (probs[:,1] >= threshold).astype(int)

        predictions = pd.Series(preds, index=Yt.index)

        return {'predictions':predictions, 'threshold':threshold}
    
    def save(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

        filename = path + self.name + '_XGB.model'
        self.model.save_model(filename)

        return f'Model saved as {filename}.'
    
    def load(self, path):

        filename = path + self.name + '_XGB.model'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        else:
            booster = xgb.Booster()
            model = booster.load_model(filename) 

        return model
    
class Regression_XGBoost(Model):
    """
    This model uses XGBClassifier to predict if a security will go up or down tomorrow depending on the data today. 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create model
        self.model = xgb.XGBRegressor( n_estimators=self.n_estimators, eta=self.learning_rate, 
                                        objective='reg:squarederror', eval_metric=['rmse', 'mae'], reg_lambda=1.0)
    
    def train(self, X, Y, Xv, Yv):
        
        if self.wandb: run = wandb_setup(**self.kwargs)

        # wandb compatibility
        self.callback = None if not self.wandb else [XGBoostWandbCallback(log_model=True)]

        self.model.fit(X, Y, eval_set=[(X, Y),(Xv, Yv)],
                    verbose=100,
                    callbacks=self.callback)
        
        if self.wandb: run.finish()

        return
        
    def predict(self, Xt, Yt, threshold=0.5):
        preds = self.model.predict(Xt)

        predictions = pd.Series(preds, index=Yt.index)

        return {'predictions':predictions}
    
    def save(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

        filename = path + self.name + '_XGB.model'
        self.model.save_model(filename)

        return f'Model saved as {filename}.'
    
    def load(self, path):

        filename = path + self.name + '_XGB.model'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        else:
            booster = xgb.Booster()
            model = booster.load_model(filename) 

        return model
                    
                    
class Binary_LSTM(Model):
    """
    This model uses a RNN with LSTM cell/s to predict if a security will go up or down tomorrow 
    depending on the data today. There are 3 models specified by their capacity (complexity). 
    """
    def __init__(self, capacity='low', **kwargs):
        super().__init__(**kwargs)
        self.capacity = capacity
        
        if self.capacity == 'low':
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(self.lstm_units, return_sequences=False, input_shape=self.input_shape),
                tf.keras.layers.Dense(1, activation='sigmoid')])

        elif self.capacity == 'medium':
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, input_shape=self.input_shape),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.dense_units, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')])

        elif self.capacity == 'high':
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, input_shape=self.input_shape),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.LSTM(self.lstm_units, return_sequences=False),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.dense_units, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')])
            
        if self.optimizer == "Adam":
            self.optim = tf.keras.optimizers.Adam(learning_rate=self.learning_rate) 
        elif self.optimizer == "SGD": 
            self.optim = tf.keras.optimizers.SGD(learning_rate=self.learning_rate) 
        elif self.optimizer == "RMSprop": 
            self.optim = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate) 
        
        self.model.compile(optimizer=self.optim, loss='binary_crossentropy', metrics=['accuracy'])
        
    def train(self, X, Y, Xv, Yv):

        # transform features for LSTM use
        X = transform_features_LSTM(X)
        Xv = transform_features_LSTM(Xv)

        if self.wandb: run = wandb_setup(**self.kwargs)

        # wandb compatibility
        self.callback = None if not self.wandb else [KerasWandbCallback()]

        history = self.model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, 
                validation_data=(Xv, Yv), callbacks=self.callback)
        
        
        if self.wandb: run.finish()

        return history
    
    def predict(self, Xt, Yt, threshold=0.5):

        Xt = transform_features_LSTM(Xt)

        probs = self.model.predict(Xt)
        preds = (probs >= threshold).astype(int).flatten()

        predictions = pd.Series(preds, index=Yt.index)

        return {'predictions':predictions, 'threshold':threshold}
    
    def save(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

        filename = path + self.name + '_LSTM.h5'
        self.model.save(filename)

        return f'Model saved as {filename}.'
    
    def load(self, path):

        filename = path + self.name + '_LSTM.h5'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        else:
            model = tf.keras.models.load_model(filename) 

        return model
    
class Regression_LSTM(Model):
    """
    This model uses a RNN with LSTM cell/s to predict if a security will go up or down tomorrow 
    depending on the data today. There are 3 models specified by their capacity (complexity). 
    """
    def __init__(self, capacity='low', **kwargs):
        super().__init__(**kwargs)
        self.capacity = capacity
        
        if self.capacity == 'low':
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(self.lstm_units, return_sequences=False, input_shape=self.input_shape),
                tf.keras.layers.Dense(1, activation='linear')])

        elif self.capacity == 'medium':
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(self.lstm_units, return_sequences=False, input_shape=self.input_shape),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.dense_units, activation='relu'),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(1, activation='linear')])

        elif self.capacity == 'high':
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, input_shape=self.input_shape),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.LSTM(self.lstm_units, return_sequences=False),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(1, activation='linear')])
            
        if self.optimizer == "Adam":
            self.optim = tf.keras.optimizers.Adam(learning_rate=self.learning_rate) 
        elif self.optimizer == "SGD": 
            self.optim = tf.keras.optimizers.SGD(learning_rate=self.learning_rate) 
        elif self.optimizer == "RMSprop": 
            self.optim = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate) 
        
        self.model.compile(optimizer=self.optim, loss='mean_squared_error', metrics=['mae'])
        
    def train(self, X, Y, Xv, Yv):

        # transform features for LSTM use
        X = transform_features_LSTM(X)
        Xv = transform_features_LSTM(Xv)

        if self.wandb: run = wandb_setup(**self.kwargs)

        # wandb compatibility
        self.callback = None if not self.wandb else [KerasWandbCallback()]

        history = self.model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, 
                validation_data=(Xv, Yv), callbacks=self.callback)
        
        
        if self.wandb: run.finish()

        return history
    
    def predict(self, Xt, Yt, threshold=0.5):

        Xt = transform_features_LSTM(Xt)

        preds = self.model.predict(Xt).flatten()
        # preds = (probs >= threshold).astype(int).flatten()

        predictions = pd.Series(preds, index=Yt.index)

        return {'predictions':predictions}
    
    def save(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

        filename = path + self.name + '_LSTM.h5'
        self.model.save(filename)

        return f'Model saved as {filename}.'
    
    def load(self, path):

        filename = path + self.name + '_LSTM.h5'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        else:
            model = tf.keras.models.load_model(filename) 

        return model
    
class Binary_RNN(Model):
    """
    This model uses a RNN to predict if a security will go up or down tomorrow depending
    on the data today. There are 3 models specified by their capacity (complexity). 
    """
    def __init__(self, capacity='low', **kwargs):
        super().__init__(**kwargs)
        self.capacity = capacity

        if self.capacity == 'low':  
            self.model = tf.keras.Sequential([
                tf.keras.layers.SimpleRNN(self.dense_units, return_sequences=True, input_shape=self.input_shape),
                tf.keras.layers.Dense(1, activation='sigmoid')])
            
        elif self.capacity == 'medium':  
            self.model = tf.keras.Sequential([
                tf.keras.layers.SimpleRNN(self.dense_units, return_sequences=True, input_shape=self.input_shape),
                tf.keras.layers.Dense(self.dense_units, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')])
            
        elif self.capacity == 'high':  
            self.model = tf.keras.Sequential([
                tf.keras.layers.SimpleRNN(self.dense_units, return_sequences=True, input_shape=self.input_shape),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.SimpleRNN(self.dense_units, return_sequences=True),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(1, activation='sigmoid')])
            
        if self.optimizer == "Adam":
            self.optim = tf.keras.optimizers.Adam(learning_rate=self.learning_rate) 
        elif self.optimizer == "SGD": 
            self.optim = tf.keras.optimizers.SGD(learning_rate=self.learning_rate) 
        elif self.optimizer == "RMSprop": 
            self.optim = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate) 
        
        self.model.compile(optimizer=self.optim, loss='binary_crossentropy', metrics=['accuracy'])
        
    def train(self, X, Y, Xv, Yv):

        # transform features for LSTM use
        X = transform_features_LSTM(X)
        Xv = transform_features_LSTM(Xv)

        if self.wandb: run = wandb_setup(**self.kwargs)

        # wandb compatibility
        self.callback = None if not self.wandb else [KerasWandbCallback()]

        history = self.model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, 
                validation_data=(Xv, Yv), callbacks=self.callback)
        
        
        if self.wandb: run.finish()

        return history
    
    def predict(self, Xt, Yt, threshold=0.5):

        Xt = transform_features_LSTM(Xt)
        
        probs = self.model.predict(Xt)
        preds = (probs >= threshold).astype(int).flatten()

        predictions = pd.Series(preds, index=Yt.index)

        return {'predictions':predictions, 'threshold':threshold}
    
    def save(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

        filename = path + self.name + '_RNN.h5'
        self.model.save(filename)

        return f'Model saved as {filename}.'
    
    def load(self, path):

        filename = path + self.name + '_RNN.h5'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        else:
            model = tf.keras.models.load_model(filename) 

        return model
    
class Binary_CNN(Model):
    """
    This model uses a CNN to predict if a security will go up or down tomorrow 
    depending on the data today. 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = 3
        self.filters = 64

          # class binary_CNN(tf.keras.Model):
        #     def __init__(self, input_shape, num_filters=64, kernel_size=3, dropout_rate=0.2):
        #         super().__init__()
        #         self.conv1d = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape)
        #         self.flatten = tf.keras.layers.Flatten()
        #         self.dropout = tf.keras.layers.Dropout(dropout_rate)
        #         self.out = tf.keras.layers.Dense(1, activation='sigmoid')

        #     def call(self, inputs):
        #         x = self.conv1d(inputs)
        #         x = self.flatten(x)
        #         x = self.dropout(x)
        #         return self.out(x)
            
        # self.model = binary_CNN(input_shape=self.input_shape)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=self.filters, 
                                   kernel_size=self.kernel_size, 
                                   activation='relu', 
                                   padding='same', 
                                   input_shape=self.input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1, activation='sigmoid')])

        if self.optimizer == "Adam":
            self.optim = tf.keras.optimizers.Adam(learning_rate=self.learning_rate) 
        elif self.optimizer == "SGD": 
            self.optim = tf.keras.optimizers.SGD(learning_rate=self.learning_rate) 
        elif self.optimizer == "RMSprop": 
            self.optim = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate) 
        
        self.model.compile(optimizer=self.optim, loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X, Y, Xv, Yv):

        # transform features for CNN using non-overlapping windows
        X, Y = transform_features_CNN_overlapping(X, Y)
        Xv, Yv = transform_features_CNN_overlapping(Xv, Yv)

        if self.wandb: run = wandb_setup(**self.kwargs)

        # wandb compatibility
        self.callback = None if not self.wandb else [KerasWandbCallback()]

        history = self.model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, 
                validation_data=(Xv, Yv), callbacks=self.callback)
        
        
        if self.wandb: run.finish()

        return history
    
    def predict(self, Xt, Yt, threshold=0.5):

        Xt, _ = transform_features_CNN_overlapping(Xt, Yt) # redefining Yt will change the index
        
        probs = self.model.predict(Xt)
        preds = (probs >= threshold).astype(int).flatten()

        # this line does not work with the rolling windowns of CNN...
        # predictions = pd.Series(preds, index=Yt.index)

        return {'predictions':preds, 'threshold':threshold}
    
    def save(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

        filename = path + self.name + '_CNN.h5'
        self.model.save(filename)

        return f'Model saved as {filename}.'
    
    def load(self, path):

        filename = path + self.name + '_CNN.h5'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        else:
            model = tf.keras.models.load_model(filename) 

        return model

####################################################
############### SPLITTING FUNCTIONS ################
####################################################
def held_out(features, target, test_size=10, **kwargs):
    """
    Reserves a number of data for testing 
    Args:
        features (dataframe) : dataframe of all the test features
        target (dataframe) : dataframe for the target 
    Kwargs:
        test_size (int) : test data split from the full dataset to do predictions on. Number is the number of
        days/periods to predict out to
        **kwargs : kwargs for the train_test_split function from sklearn
    Returns:
        
    """  
    X = features.copy()
    Y = target.copy()

    # training and validation data
    features, target = X[:-test_size], Y[:-test_size]
    # held out dataset for testing
    X_test, Y_test = X[-test_size:], Y[-test_size:]

    return features, target, X_test, Y_test

def val_split(features, target, test_size=0.2, **kwargs):
    """
    Splits the data into training and validation 
    Args:
        features (dataframe) : dataframe of all the test features
        target (dataframe) : dataframe for the target 
    Kwargs:
        test_size (float) : validation data split from full data set, i.e. 0.2 -> 20% test to training data
        **kwargs : kwargs for the train_test_split function from sklearn
    Returns:
        A dictionary with the train and test arrays
    """    

    x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=test_size, **kwargs) 

    return {'X_train': x_train, 'Y_train': y_train, 'X_val': x_val, 'Y_val': y_val}

def binary_cross_validation(model, features, target, n_folds=5, val_size=None, test_period=10, **kwargs):
    wandb_flag = kwargs['wandb']

    if wandb_flag: 
        name = kwargs['name']
        name += '_cross_validation'
        kwargs['name'] = name

        run = wandb_setup(kwargs)

    # create splitting object 
    tss = TimeSeriesSplit(n_splits=n_folds, test_size=val_size)

    cv_scores = cross_val_score(model, features, target, cv=tss, scoring='accuracy', verbose=100)

    # cross-validation scores
    print("Cross-Validation Scores:", cv_scores)

    # mean and standard deviation of the scores
    print(f"Mean Accuracy: {cv_scores.mean():.2f}")
    print(f"Standard Deviation: {cv_scores.std():.2f}")

    if wandb_flag: run.finish()

    return

####################################################
################## WANDB FUNCTIONS #################
####################################################
def wandb_setup(**kwargs):

    # log in 
    wandb.login(key=API_KEY)

    # create run
    run = wandb.init(
        # Set the project where this run will be logged
        project=kwargs['name'],
        name=kwargs['run'],
        entity=kwargs['entity'],
        # Track hyperparameters and run metadata
        config={
        'learning_rate': kwargs['learning_rate'],
        'model_architecture': kwargs['model_architecture'],
        'objective':kwargs['objective'],
        'eval_metrics':['logloss', 'error'],
        'dataset': kwargs['dataset'],
        'epochs': kwargs['epochs'],
        'optimizer':kwargs['optimizer'],
        'test_split':kwargs['test_split'],
        'validation_split':kwargs['validation_split']
        })
    
    return run

####################################################
################ ANALYSIS FUNCTIONS ################
####################################################
def error_matrix(truth, prediction, save=False, **kwargs):

    conf_matrix = confusion_matrix(truth, prediction)
    errors = error_metrics(truth, prediction)
    accuracy = errors['accuracy']
    precision = errors['precision']
    recall = errors['recall']
    f1 = errors['f1']
    specificity = errors['specificity']


    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    
    metrics_text = f'Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}\nSpecificity: {specificity:.2f}'
    plt.text(1.7, 0.35, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=1))
    
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    if save: 
        name = kwargs['name']
        plt.savefig(name + 'confusion_matrix.pdf')
    
    plt.show()

def error_metrics(truth, prediction):

    conf_matrix = confusion_matrix(truth, prediction)

    TN = conf_matrix[0,0]
    TP = conf_matrix[1,1]
    FN = conf_matrix[1,0]
    FP = conf_matrix[0,1]

    recall = TP / (TP+FN) # correctly identify positives
    precision = TP / (TP+FP) # positive only
    specificity = TN / (TN+FP) # negative only
    f1 = 2 * (precision*recall) / (precision+recall) # total 
    accuracy = np.isclose(truth, prediction).sum() / len(truth)

    return {'recall':recall, 'precision':precision, 'specificity':specificity, 'f1':f1, 'accuracy':accuracy}



def plot(history):
    sns.set(style="whitegrid", palette="muted")
    # Create a figure and axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot training/test loss and accuracy
    sns.lineplot(x=range(1, len(history.history['accuracy']) + 1), y=history.history['accuracy'], ax=axes[1], label='Training Accuracy')
    sns.lineplot(x=range(1, len(history.history['val_accuracy']) + 1), y=history.history['val_accuracy'], ax=axes[1], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(loc='lower right')

    sns.lineplot(x=range(1, len(history.history['loss']) + 1), y=history.history['loss'], ax=axes[0], label='Training Loss')
    sns.lineplot(x=range(1, len(history.history['val_loss']) + 1), y=history.history['val_loss'], ax=axes[0], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(loc='upper right')

    plt.tight_layout()

    # Show the plot
    plt.show()

    return

# NOTE: Adding this to avoid having to transform the freautes before I pass it to the model. 
# In order to do this you need to first add the training/test split to this module. 
# Use the dictionary keywords to determine the split and pass the features directly into the models. 
def transform_features_LSTM(df, period=1):
    """
    Args:
        df (dataframe) : the FEATURES data frame
    Kwargs:
        period (int) : period of sampling for LSTM later
    Returns:
        A numpy array with the correct 3D dimensions for the LSTM machine learning model
    """
    df = df.copy()
    samples, features = df.shape
    npdf = df.to_numpy()
    npdf = npdf.reshape((samples, period, features))

    return npdf

def transform_features_CNN(features, target, window=10):

    # combine dataframes
    data = pd.concat([features, target], axis=1)
    
    # non-overlapping windows
    X = []
    Y = []

    for i in range(0, len(data) - window + 1, window):
        X.append(data.iloc[i:i + window, :-1].values)  # features for each window
        Y.append(data.iloc[i + window - 1, -1])  # target value for each window

    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    return (X, Y)

def transform_features_CNN_overlapping(features, target, window=10, step=1):
    # Combine dataframes
    data = pd.concat([features, target], axis=1)

    # Overlapping windows
    X = []
    Y = []

    for i in range(0, len(data) - window + 1, step):
        X.append(data.iloc[i:i + window, :-1].values)  # Features for each window
        Y.append(data.iloc[i + window - 1, -1])  # Target value for each window

    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    return X, Y