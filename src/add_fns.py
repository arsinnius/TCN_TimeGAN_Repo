"""
add_qtrs
load_pkl
make_nlreg_nn
make_random_data
make_rnn
make_vars_dict
save_pkl
"""

from tcn_timegan.src.params import latent_dim, n_seq, seq_len

from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import os, pickle
import pandas as pd


def add_qtrs(date, qtrs):
    """
    Input
        date:   string or datetime
        qtrs:   number of quarters to be added or subtracted
        
    Output  date +/- offset
    """
    ts = pd.Timestamp(date) + pd.DateOffset(months=3 * qtrs)
    per = ts.to_period('Q')
    return per.strftime('%YQ%q')


def make_random_data():
    while True:
        yield np.random.uniform(low=0, high=1, size=(seq_len, n_seq))


def make_rnn(n_layers, input_dim, hidden_units, output_units, name):
    return Sequential([Input(shape=[input_dim, latent_dim])] +
                      [GRU(units=hidden_units,
                           return_sequences=True,
                           name=f'GRU_{i + 1}') for i in range(n_layers)] +
                      [Dense(units=output_units,
                             activation='sigmoid',
                             name='OUT')], name=name)  


def make_nlreg_nn(input_dim, loss='mean_squared_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200)):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim, name='Dense1'))
    model.add(Dense(32, activation="relu", name='Dense2'))
    model.add(Dense(8, activation="relu", name='Dense3'))
    
    # Since the regression is performed, a Dense layer containing a single neuron with a linear activation function.
    # Typically ReLu-based activations are used but since this is a regression, a linear activation is used.
    model.add(Dense(1, activation="linear", name='Dense4'))
    
    # Compile model: The model is initialized with the Adam optimizer and then it is compiled.
    model.compile(loss=loss, optimizer=optimizer)
    return model


def load_pkl(filename, data_path):
    fname = os.path.join(data_path, filename + '.pkl')
    with open(fname, 'rb') as input:
        obj = pickle.load(input)
    return obj 

    
def save_pkl(obj, filename, data_path):
    filename = os.path.join(data_path,(filename + '.pkl'))
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL) 
 
    
def make_vars_dict(fit_vars_dict, start_time, end_time, var_tofit_dict=None ):
    """
    Create dictionary holding the fitting data. For fitting (nonlinear regression),
    the variable to be fit will be first.
    
    Input
        fit_vars_dict:   dictionary of fitting variables
        start_time: start time of variabe to be fit, e.g. '1981Q1' for WILSH5K
        end_time: common end_time for all variables, e.g. '2021Q3' for WILSH5K
        var_tofit_dict: {nameof(var_tofit) : var_tofit}  or None
    Output
        df: dataframe holding the fitting variables 
    """
    
    if var_tofit_dict != None:
        # Make first key-val pair the variable to be fit
        vars_dict = var_tofit_dict
        vars_dict.update(fit_vars_dict)
    else:
        vars_dict = fit_vars_dict

    for var_name, var_val in vars_dict.items():
      vars_dict[var_name] = var_val.loc[start_time:end_time, :]
    return vars_dict        
 
    