from numpy.lib.function_base import average
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class GeneralANN:
    raw_data = pd.DataFrame()
    train_inp = pd.DataFrame()
    train_out = pd.DataFrame()
    test_inp = pd.DataFrame()
    test_out = pd.DataFrame()
    fraction = 0.0
    layers = []
    activation_functions = []
    normalizer = 0
    model_history = 0
    model = 0

    def define_layers(self, layers, functions):
        for layer, function in zip(layers, functions):
            self.layers.append(layer)
            self.activation_functions.append(function)

    def prepare_data(self, raw_data, inp, out, fraction=0.8):
        self.fraction = fraction
        dataset = raw_data.copy()
        self.raw_data = dataset
        train = dataset.sample(frac=fraction, random_state=0)
        self.train_inp = train[inp]
        self.train_out = train[out]
        test = dataset.drop(train.index)
        self.test_inp = test[inp]
        self.test_out = test[out]
        

    def normalize_data(self, show_example = False):

        self.normalizer = preprocessing.Normalization()
        self.normalizer.adapt(np.array(self.train_inp))
        first = np.array(self.train_inp[:1])

        if show_example:
            with np.printoptions(precision=2, suppress=True):
                print('First example:', first)
                print('Normalized:', self.normalizer(first).numpy())
    
    def build_and_compile_model(self, ann_layers, functions, eta=0.0001):
        if len(ann_layers)-1 != len(functions): 
            print("wrong number of layers or activation functions!")
            print(ann_layers)
            print(functions)
        arguments = [self.normalizer]
        for layer, function in zip(ann_layers[:-1], functions):
                arguments.append(layers.Dense(layer, activation=function))
        arguments.append(layers.Dense(ann_layers[-1]))
        
        self.model = keras.Sequential(arguments)
        self.model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(eta))

    def plot_loss(self):
        plt.plot(self.model_history.history['loss'], label='loss')
        y2 = max(self.model_history.history['val_loss'])
        y1 = min(self.model_history.history['val_loss'])
        plt.plot(self.model_history.history['val_loss'], label='val_loss')
        plt.ylim([0.8*y1, 1.1*y2])
        plt.xlabel('Epoch')
        plt.ylabel('Error [rho]')
        plt.legend()
        plt.grid(True)
        plt.show()

    def print_weights(self):
        i = 0
        for layer in self.model.layers[1:]:
            i+=1
            for each in layer.get_weights()[0][:]:
                print("layer ",i, each)

    def plot_scheme(self):
        plot_model(self.model, to_file='model.png', show_shapes=True, show_dtype=False,
                    show_layer_names=False, rankdir='TB', expand_nested=True, dpi=96)
   
    def train(self, batch = 10, epochs = 100):
        self.model_history = self.model.fit(self.train_inp, self.train_out, 
                validation_split=0.2, batch_size=batch,
                verbose=2, epochs=epochs)

    def test(self, test_inp=0, test_out=0):
        print("**********************************************************************")
        if test_inp==0:
            test_inp = self.test_inp
            test_out = self.test_out
            print("test on the %.2f %% of data, unused for training" %((1-self.fraction)*100))
        test_results = self.model.evaluate(test_inp, test_out, verbose=1)
        print("test result: ", test_results)
        print("**********************************************************************")

np.set_printoptions(precision=2, suppress=True)
raw_dataset = pd.read_csv("data_RM.csv",
                          na_values=' ', comment='\t',
                          sep=',', skipinitialspace=True)

