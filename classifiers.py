import warnings
import os
import json

import pandas as pd
import numpy as np
import tensorflow as tf

from joblib import dump, load
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping

print(f"TensorFlow version: {tf.__version__}")

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Configuration():
    # Contains everything we need to make an experimentation

    def __init__(
        self,
        max_length = 150,
        padding = True,
        batch_size = 32,
        epochs = 50,
        learning_rate = 1e-5,
        metrics = ["accuracy"],
        verbose = 1,
        split_size = 0.25,
        accelerator = "TPU",
        myluckynumber = 13,
        first_time = True,
        save_model = True
    ):
        # seed and accelerator
        self.SEED = myluckynumber
        self.ACCELERATOR = accelerator

        # save and load parameters
        self.FIRST_TIME = first_time
        self.SAVE_MODEL = save_model

        #Data Path
        self.DATA_PATH = Path('dataset.csv')
        self.EMBEDDING_INDEX_PATH = Path('fr_word.vec')

        # split
        self.SPLIT_SIZE = split_size

        # model hyperparameters
        self.MAX_LENGTH = max_length
        self.PAD_TO_MAX_LENGTH = padding
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.METRICS = metrics
        self.VERBOSE = verbose

        # initializing accelerator
        self.initialize_accelerator()

    def initialize_accelerator(self):

        #Initializing accelerator

        # checking TPU first
        if self.ACCELERATOR == "TPU":
            print("Connecting to TPU")
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                print(f"Running on TPU {tpu.master()}")
            except ValueError:
                print("Could not connect to TPU")
                tpu = None

            if tpu:
                try:
                    print("Initializing TPU")
                    tf.config.experimental_connect_to_cluster(tpu)
                    tf.tpu.experimental.initialize_tpu_system(tpu)
                    self.strategy = tf.distribute.TPUStrategy(tpu)
                    self.tpu = tpu
                    print("TPU initialized")
                except _:
                    print("Failed to initialize TPU")
            else:
                print("Unable to initialize TPU")
                self.ACCELERATOR = "GPU"

        # default for CPU and GPU
        if self.ACCELERATOR != "TPU":
            print("Using default strategy for CPU and single GPU")
            self.strategy = tf.distribute.get_strategy()

        # checking GPUs
        if self.ACCELERATOR == "GPU":
            print(f"GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

        # defining replicas
        self.AUTO = tf.data.experimental.AUTOTUNE
        self.REPLICAS = self.strategy.num_replicas_in_sync
        print(f"REPLICAS: {self.REPLICAS}")


def TFIDF_vectorizer(x_train, x_test, first_time):
    # Used for logistic regression
    if first_time:
        print('Building TF-IDF Vectorizer')
        vectorizer = TfidfVectorizer(ngram_range = (1,4))
        vectorizer.fit(x_train)
        dump(vectorizer, 'tfidf_vectorizer.joblib', compress= 3)
    else:
        print('Loading our TF-IDF vectorizer')
        vectorizer = load('tfidf_vectorizer.joblib')
    print('Vectorizing our sequences')
    x_train, x_test = vectorizer.transform(x_train), vectorizer.transform(x_test)
    print('Data Vectorized')
    return x_train, x_test

def load_embedding_index(file_path):
    embedding_index = {}
    for _, line in enumerate(open(file_path)):
        values = line.split()
        embedding_index[values[0]] = np.asarray(
                            values[1:], dtype='float32')
    return embedding_index

def build_embedding_matrix(x_train, x_test, maxlen, first_time, file_path):

    #Tokenizer
    if first_time :
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(x_train)
        dump(tokenizer, 'tokenizer.joblib', compress= 3)
    else:
        tokenizer = load('tokenizer.joblib')

    #Word index
    word_index = tokenizer.word_index

    #Embedding matrix
    if first_time:
        print('Loading embedding index')
        embedding_index = load_embedding_index(file_path)
        print('Building our embedding matrix')
        embedding_matrix = np.zeros(
            (len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        dump(embedding_matrix, 'embedding_matrix.joblib', compress= 3)
    else:
        embedding_matrix = load('embedding_matrix.joblib')

    # Tokenzing + padding
    seq_x_train = sequence.pad_sequences(
            tokenizer.texts_to_sequences(x_train), maxlen=maxlen)

    seq_x_test = sequence.pad_sequences(
            tokenizer.texts_to_sequences(x_test), maxlen=maxlen)

    return seq_x_train, seq_x_test, embedding_matrix, word_index

def build_LogisticRegression(x_train, y_train, save_model, C=110):
    print('Fitting Logistic Regression')
    modelLR = LogisticRegression(C= C, max_iter=300)
    modelLR.fit(x_train, y_train)
    print('Logistic Regression fitted')

    if save_model:
        print('Saving model')
        dump(modelLR, 'modelLR.joblib', compress = 3)
    return modelLR

def build_RandomFR(x_train, y_train, save_model):
    print('Fitting our Random Forest')
    modelRF = RandomForestClassifier(n_estimators =100).fit(x_train, y_train)
    print('Random Forest Fitted')
    if save_model:
        print('Saving model')
        dump(modelRF, 'modelRF.joblib', compress = 3)
    return modelRF

def build_LSTM(embedding_matrix, word_index, maxlen, learning_rate, metrics, first_time):

    input_strings = Input(shape=(maxlen,))

    x = Embedding(len(word_index) + 1, 300, input_length=maxlen,
                        weights=[embedding_matrix],
                        trainable=False)(input_strings)
    x = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(x)
    x= Dense(1, activation="sigmoid")(x)

    model = Model(inputs = input_strings, outputs = x)

    opt = Adam(learning_rate = learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()

    model.compile(optimizer= opt, loss= loss, metrics = metrics)

    if not first_time:
        model.load_weights("lstm_model.h5")

    return model

def get_tf_dataset(X, y, auto, labelled = True, repeat = False, shuffle = False, batch_size = 32):
    """
    Creating tf.data.Dataset for TPU.
    """
    if labelled:
        ds = (tf.data.Dataset.from_tensor_slices((X, y)))
    else:
        ds = (tf.data.Dataset.from_tensor_slices(X))

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(2048)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(auto)

    return ds

def run_LogisticRegression(config):
    # Reading data
    data = pd.read_csv(config.DATA_PATH)

    #separating sentences and labels
    sentences = data.text.astype(str).values.tolist()
    labels = data.label.astype(float).values.tolist()

    # splitting data into training and validation
    X_train, X_valid, y_train, y_valid = train_test_split(sentences,
                                                          labels,
                                                          test_size = config.SPLIT_SIZE
                                                          )

    #Vectorizing data
    X_train, X_valid = TFIDF_vectorizer(X_train, X_valid, config.FIRST_TIME)

    #Building model
    model = build_LogisticRegression(X_train, y_train, save_model = config.SAVE_MODEL)

    #predicting outcomes
    y_pred = model.predict(X_valid)

    print(classification_report(y_valid, y_pred))

def run_RandomForest(config):
    # Reading data
    data = pd.read_csv(config.DATA_PATH)

    #separating sentences and labels
    sentences = data.text.astype(str).values.tolist()
    labels = data.label.astype(float).values.tolist()

    # splitting data into training and validation
    X_train, X_valid, y_train, y_valid = train_test_split(sentences,
                                                          labels,
                                                          test_size = config.SPLIT_SIZE
                                                          )

    #Vectorizing data
    X_train, X_valid = TFIDF_vectorizer(X_train, X_valid, config.FIRST_TIME)

    #Building model
    model = build_RandomFR(X_train, y_train, save_model = config.SAVE_MODEL)

    #predicting outcomes
    y_pred = model.predict(X_valid)

    print(classification_report(y_valid, y_pred))

def run_lstm_model(config):
    """
    Run model
    """
    # Reading data
    data = pd.read_csv(config.DATA_PATH)

    #separating sentences and labels
    sentences = data.text.astype(str).values.tolist()
    labels = data.label.astype(float).values.tolist()

    # splitting data into training and validation
    X_train, X_valid, y_train, y_valid = train_test_split(sentences,
                                                          labels,
                                                          test_size = config.SPLIT_SIZE
                                                          )

    # Building embedding word to vector:
    seq_x_train, seq_x_test, embedding_matrix, word_index = build_embedding_matrix(
                                                                            X_train,
                                                                            X_valid,
                                                                            config.MAX_LENGTH,
                                                                            config.FIRST_TIME,
                                                                            config.EMBEDDING_INDEX_PATH)

    # initializing TPU
    #if config.ACCELERATOR == "TPU":
        #if config.tpu:
            #config.initialize_accelerator()


    # building model
    K.clear_session()
    #with config.strategy.scope(): (doesn't work because of our embedding layer, has to be fixed)
    model = build_LSTM(embedding_matrix, word_index, config.MAX_LENGTH, config.LEARNING_RATE, config.METRICS, config.FIRST_TIME)

    print('model builded')

     # creating TF Dataset (not used since multiprocessing doesn't work with our embedding model)
    #ds_train = get_tf_dataset(X_train, y_train, config.AUTO, repeat = True, shuffle = True, batch_size = config.BATCH_SIZE * config.REPLICAS)
    #ds_valid = get_tf_dataset(X_valid, y_valid, config.AUTO, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)

    n_train = len(X_train)

     # saving model at best accuracy epoch
    sv = [tf.keras.callbacks.ModelCheckpoint(
        "lstm_model.h5",
        monitor = "val_accuracy",
        verbose = 1,
        save_best_only = True,
        save_weights_only = True,
        mode = "max",
        save_freq = "epoch"),
        tf.keras.callbacks.EarlyStopping(patience = 10,
                                         verbose= 1,
                                         monitor='val_accuracy')]
    print("\nTraining")

     # training model
    seq_x_train = np.array(seq_x_train)
    y_train = np.array(y_train)
    seq_x_test = np.array(seq_x_test)
    y_valid = np.array(y_valid)

    model_history = model.fit(
        x = seq_x_train,
        y = y_train,
        epochs = config.EPOCHS,
        callbacks = [sv],
        batch_size = config.BATCH_SIZE,
        #steps_per_epoch = n_train / config.BATCH_SIZE // config.REPLICAS,
        validation_data = (seq_x_test, y_valid),
        verbose = config.VERBOSE
        )

    print("\nValidating")

    # scoring validation data
    model.load_weights("lstm_model.h5")

    #ds_valid = get_tf_dataset(X_valid, -1, config.AUTO, labelled = False, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)
    preds_valid = model.predict(seq_x_test, verbose = config.VERBOSE)

    print('Classification report:')
    print(classification_report(y_valid, (preds_valid > 0.5)))

    if config.SAVE_MODEL:
        model_json = model.to_json()
        json.dump(model_json, 'lstm_model.json')
