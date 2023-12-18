import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, BatchNormalization
from keras.optimizers import Adam

from tensorflow.keras.layers import Concatenate, Permute, Dot, Input, Multiply
from tensorflow.keras.layers import RepeatVector, Activation, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from babel.dates import format_date
import matplotlib.pyplot as plt
from keras.initializers import Initializer

from tensorflow.keras.activations import softmax
from tensorflow.keras.initializers import GlorotUniform

import warnings
warnings.filterwarnings('ignore')

class data_process(object):
    def __init__(self,config,data):
        self.data = data
        self.Tx = config.Tx
        self.experiment_type = config.experiment_type
    def prepare_data(self):
        data = self.data
        m_r=max(data['reads'])
        m_s=max(data['signal_to_noise'])
        
        # Fill NaN values in 'reads' column with 0
        data['reads'].fillna(0, inplace=True)
        
        # Fill NaN values in 'signal_to_noise' column with 0
        data['signal_to_noise'].fillna(0, inplace=True)
        #Filtering data with SN=0, below reads and snr threshold and experiment type
        data = data[data['SN_filter'] != 0]
        data2 = data[(data['reads'] >= 0.001 * m_r) & (data['signal_to_noise'] >= 0.01 * m_s)]
        data = data2[data2['experiment_type'] != self.experiment_type]
        # Padding the sequences with a specific character, say 'N'
        max_sequence_length = self.Tx
        
        # Pad sequences by repeating the start
        data['padded_sequence'] = data['sequence'].apply(lambda x: (x * (max_sequence_length // len(x) + 1))[:max_sequence_length])
        #data['padded_sequence'] = data.apply(lambda row: row['sequence'] + row['sequence'][:457 - len(row['sequence'])], axis=1)
        # Encode sequences to numerical format
        encoder = LabelEncoder()
        data['sequence_encoded'] = data['padded_sequence'].apply(lambda x: encoder.fit_transform(list(x)))
        
        # Convert the encoded sequences to a matrix format
        X = np.array(data['sequence_encoded'].tolist())
        L= np.array(data['sequence_encoded'].tolist()).shape[1]
        #y = data['reactivity_0055'].isna()
        #print(y)
        #a=random.randint(1, 1000000)
        y = data.iloc[:, 7:213].values
        nan_indices = np.isnan(y)
        Y =y
        Y= np.array([np.pad(element, (0, max_sequence_length - len(element)), 'constant', constant_values=0) for element in y])
        # Iterate through each NaN index and assign a random number
        for i, j in zip(*np.where(nan_indices)):
            Y[i, j] = random.uniform(-0.1, 0.1)
            
        #Y= [np.concatenate([x, x[:1]])[:max_sequence_length] for x in Y]
        
        # Reshaping the input for LSTM model
        
        
        X = X.reshape(X.shape[0], X.shape[1], 1)
        X=X[:,:]
        one_hot_X= tf.keras.utils.to_categorical(X, num_classes=np.max(X)+1)
        return one_hot_X, Y
    
class train(object):
    def __init__(self,config):
        self.Tx = config.Tx
        self.n_a = config.n_a
        self.n_s = config.n_s
        self.input_size = config.input_size
    
    def one_step_attention(self, a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        
        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        
        Returns:
        context -- context vector, input of the next (post-attention) LSTM cell
        """
        
        repeator = RepeatVector(self.Tx)
        concatenator = Concatenate(axis=-1)
        densor1 = Dense(3, activation="tanh", kernel_initializer=GlorotUniform())
        densor2 = Dense(1, activation="relu", kernel_initializer=GlorotUniform())
        activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
        dotor = Dot(axes = 1)
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
        s_prev = repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        # For grading purposes, please list 'a' first and 's_prev' second, in this order.
        concat = concatenator([a,s_prev])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = densor1(concat)
     
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = activator(energies)
        # Use dotor together with "alphas" and "a", in this order, to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = dotor([alphas,a])
        
        return context
    
    def modelf(self):
        """
        Arguments:
        Tx -- length of the input sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        input_size -- size of the input features (number of features in RNA sequence)
    
        Returns:
        model -- Keras model instance
        """
        
        # Define the inputs of your model with shape (Tx, input_size)
        X = Input(shape=(self.Tx, self.input_size))
        
        # Initialize s0 (initial hidden state) and c0 (initial cell state)
        s0 = Input(shape=(self.n_s,), name='s0')
        c0 = Input(shape=(self.n_s,), name='c0')
        s = s0
        c = c0
        dropout_layer = Dropout(0.4)
        # Step 1: Define your pre-attention Bi-LSTM. (≈ 1 line)
        a = Bidirectional(LSTM(self.n_a, return_sequences=True))(X)
        a = BatchNormalization()(a)
        a = dropout_layer(a)
        # Step 2: Perform one step of the attention mechanism
        context = self.one_step_attention(a, s)
        context= BatchNormalization()(context)
        context = dropout_layer(context)
        # Step 3: Apply the post-attention LSTM cell to the "context" vector.
        post_activation_LSTM_cell = LSTM(self.n_s, return_state = True) # Please do not modify this global variable.
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        
        # Step 4: Apply Dense layer to the hidden state output of the post-attention LSTM
        out = Dense(self.Tx, activation='linear')(s)
        
        # Create model instance taking three inputs and returning the single output.
        model = Model(inputs=[X, s0, c0], outputs=out)
        
        return model
    