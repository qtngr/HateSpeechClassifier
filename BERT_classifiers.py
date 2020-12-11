## importing packages
import gc
import os
import random
import transformers
import warnings
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModel

print(f"TensorFlow version: {tf.__version__}")
print(f"Transformers version: {transformers.__version__}")

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## defining configuration
class Configuration_BERT():
    """
    All configuration for running an experiment
    """
    def __init__(
        self,
        model_name,
        max_length = 150,
        padding = True,
        batch_size = 32,
        epochs = 5,
        learning_rate = 1e-5,
        metrics = ["accuracy"],
        verbose = 1,
        split_size = 0.25,
        accelerator = "TPU",
        myluckynumber = 13,
        include_english = False,
        save_model = True
    ):
        # seed and accelerator
        self.SEED = myluckynumber
        self.ACCELERATOR = accelerator

        # save and load parameters
        self.SAVE_MODEL = save_model

        # english data
        self.INCLUDE_ENG = include_english

        # paths
        self.PATH_FR_DATA = Path("dataset.csv")
        self.PATH_ENG_DATA = Path("eng_dataset.csv")

        # splits
        self.SPLIT_SIZE = split_size

        # model configuration
        self.MODEL_NAME = model_name
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_NAME)

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
        
def encode_text(sequences, tokenizer, max_len, padding):
    """
    Preprocessing textual data into encoded tokens.
    """

    # encoding text using tokenizer of the model
    text_encoded = tokenizer.batch_encode_plus(
        sequences,
        pad_to_max_length = padding,
        truncation=True,
        max_length = max_len
    )

    return text_encoded

def get_tf_dataset(X, y, auto, labelled = True, repeat = False, shuffle = False, batch_size = 32):
    """
    Creating tf.data.Dataset for TPU.
    """
    if labelled:
        ds = (tf.data.Dataset.from_tensor_slices((X["input_ids"], y)))
    else:
        ds = (tf.data.Dataset.from_tensor_slices(X["input_ids"]))

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(2048)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(auto)

    return ds

## building model
def build_model(model_name, max_len, learning_rate, metrics):
    """
    Building the Deep Learning architecture
    """
    # defining encoded inputs
    input_ids = Input(shape = (max_len,), dtype = tf.int32, name = "input_ids")
    
    # defining transformer model embeddings
    transformer_model = TFAutoModel.from_pretrained(model_name)
    transformer_embeddings = transformer_model(input_ids)[0]

    # defining output layer
    output_values = Dense(512, activation = "relu")(transformer_embeddings[:, 0, :])
    output_values = Dropout(0.5)(output_values)
    #output_values = Dense(32, activation = "relu")(output_values)
    output_values = Dense(1, activation='sigmoid')(output_values)

    # defining model
    model = Model(inputs = input_ids, outputs = output_values)
    opt = Adam(learning_rate = learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = metrics

    model.compile(optimizer = opt, loss = loss, metrics = metrics)

    return model

def run_model(config):
    """
    Running the model
    """
    ## reading data
    fr_df = pd.read_csv(config.PATH_FR_DATA)

    sentences = fr_df.text.astype(str).values.tolist()
    labels = fr_df.label.astype(float).values.tolist()

    # splitting data into training and validation
    X_train, X_valid, y_train, y_valid = train_test_split(sentences,
                                                          labels,
                                                          test_size = config.SPLIT_SIZE
                                                          )
    if config.INCLUDE_ENG:
        eng_df = pd.read_csv(config.PATH_ENG_DATA)
        X_train = eng_df.text.astype(str).tolist() + X_train
        y_train = eng_df.labels.astype(float).values.tolist() + y_train
    
    # initializing TPU
    if config.ACCELERATOR == "TPU":
        if config.tpu:
            config.initialize_accelerator()

    # building model

    K.clear_session()

    with config.strategy.scope():
      model = build_model(config.MODEL_NAME, config.MAX_LENGTH, config.LEARNING_RATE, config.METRICS)
      
    #print(model.summary())

    print("\nTokenizing")

    # encoding text data using tokenizer
    X_train_encoded = encode_text(X_train, tokenizer = config.TOKENIZER, max_len = config.MAX_LENGTH, padding = config.PAD_TO_MAX_LENGTH)
    X_valid_encoded = encode_text(X_valid, tokenizer = config.TOKENIZER, max_len = config.MAX_LENGTH, padding = config.PAD_TO_MAX_LENGTH)

    # creating TF Dataset
    ds_train = get_tf_dataset(X_train_encoded, y_train, config.AUTO, repeat = True, shuffle = True, batch_size = config.BATCH_SIZE * config.REPLICAS)
    ds_valid = get_tf_dataset(X_valid_encoded, y_valid, config.AUTO, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)

    n_train = len(X_train)

    # saving model at best accuracy epoch
    sv = [tf.keras.callbacks.ModelCheckpoint(
        "model.h5",
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
    model_history = model.fit(
        ds_train,
        epochs = config.EPOCHS,
        callbacks = [sv],
        steps_per_epoch = n_train / config.BATCH_SIZE // config.REPLICAS,
        validation_data = ds_valid,
        verbose = config.VERBOSE
        )

    print("\nValidating")

    # scoring validation data
    model.load_weights("model.h5")
    ds_valid = get_tf_dataset(X_valid_encoded, -1, config.AUTO, labelled = False, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)

    preds_valid = model.predict(ds_valid, verbose = config.VERBOSE)

    print('Classification report:')
    print(classification_report(y_valid, (preds_valid > 0.5)))
