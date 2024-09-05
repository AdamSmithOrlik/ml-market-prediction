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


# from data import Data


default_hyperparameters = {
    'learning_rate': 0.01,
    'epochs': 1000,
    'batch_size': 32,
    'lstm_units': 50, 
    'dense_units': 64,
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

class Models:

    class Binary_XGBoost():
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for key, val in self.kwargs.items():
                setattr(self, key, val)

            # create model
            self.model = xgb.XGBClassifier(n_estimators=self.epochs, eta=self.learning_rate, objective='binary:logistic', 
                        eval_metric=['logloss', 'error'])
        
        def train(self, X, Y, Xv, Yv):
            
            if self.wandb: run = wandb_setup(**self.kwargs)

            # wandb compatibility
            self.callback = None if not self.wandb else [XGBoostWandbCallback(log_model=True)]

            self.model.fit(X, Y, eval_set=[(X, Y),(Xv, Yv)],
                        verbose=100,
                        callbacks=self.callback)
            
            if self.wandb: run.finish()

            return
            
        def predict(self, Xt, threshold=0.5):
            probs = self.model.predict_proba(Xt)
            preds = (probs[:,1] >= threshold).astype(int)

            predictions = pd.Series(preds, index=Xt.index)

            return {'predictions':predictions, 'threshold':threshold}
        
        def save(self, path):

            if not os.path.exists(path):
                os.makedirs(path)

            filename = path + self.name + '.model'
            self.model.save_model(filename)

            return 'Model saved.'
        
        def load(self, path):

            filename = path + self.name + '.model'
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist.")
            else:
                booster = xgb.Booster()
                model = booster.load_model(filename) 

            return model
                       
    class Binary_LSTM():
        def __init__(self, capacity='medium', **kwargs):
            self.kwargs = kwargs
            self.capacity = capacity
            for key, val in self.kwargs.items():
                setattr(self, key, val) 

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

            if self.wandb: run = wandb_setup(**self.kwargs)

            # wandb compatibility
            self.callback = None if not self.wandb else [KerasWandbCallback()]

            history = self.model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, 
                    validation_data=(Xv, Yv), callbacks=self.callback)
            
            
            if self.wandb: run.finish()

            return history
        
        def predict(self, Xt, threshold=0.5):
            probs = self.model.predict(Xt)
            preds = (probs >= threshold).astype(int).flatten()

            predictions = pd.Series(preds, index=Xt.index)

            return {'predictions':predictions, 'threshold':threshold}
        
        def plot(self, history):
            # create plot 
            plt.figure(figsize=(12, 6))

            # plot training/test loss and accuracy 
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
           
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')

            plt.show()

            return 
        


####################################################
############### SPLITTING FUNCTIONS ################
####################################################
def split(self, features, target, test_size=0.2, **kwargs):
    """
    Args:
        features (dataframe) : dataframe of all the test features
        target (dataframe) : dataframe for the target 
    Kwargs:
        test (float) : test data split from full data set, i.e. 0.2 -> 20% test to training data
        **kwargs : kwargs for the train_test_split function from sklearn
    Returns:
        A dictionary with the train and test arrays
    """    

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size, **kwargs) 

    return {'X_train': x_train, 'Y_train': y_train, 'X_test': x_test, 'Y_test': y_test}

def binary_cross_validation(self, model, features, target, n_folds=5, val_size=None, test_period=10, **kwargs):
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