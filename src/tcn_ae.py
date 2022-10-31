from tcn import TCN
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, AveragePooling1D, Conv1D,
                                     Dense, Dropout, Input, MaxPooling1D,
                                     UpSampling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
import tensorflow as tf
import time

class TCN_AE:
    """
    A class used to represent the Temporal Convolutional Encoder.

    ...

    Attributes
    ----------
    model : xxtypexx
        The TCN-AE model.

    Methods
    -------
    build_model(verbose = 1)
        Builds the model
    """
    
    model = None
    
    def __init__(self,
                 ts_len, 
                 ts_dimension,
                 latent_sample_rate,
                 pooler,
                 activation_conv1d = 'linear',
                 conv_kernel_init = 'glorot_normal',
                 dilations = (1, 2, 4, 8, 16),
                 dropout_rate = 0.00,
                 error_window_length = 128,
                 filters_conv1d = 8,        
                 kernel_size = 20,
                 learning_rate = 0.001,
                 loss = 'logcosh',
                 nb_filters = 20,
                 nb_stacks = 1,
                 padding = 'same',                                                                  
                 use_early_stopping = False,                 
                 verbose = 1
                ):
        
      
        """
        Parameters
        ----------
        ts_dimension : int
            The dimension of the time series (default is 1)
        dilations : tuple
            The dilation rates used in the TCN-AE model (default is (1, 2, 4, 8, 16))
        nb_filters : int
            The number of filters used in the dilated convolutional layers. All dilated conv. layers use the same number of filters (default is 20)
        """
        
        self.activation_conv1d = activation_conv1d
        self.conv_kernel_init = conv_kernel_init
        self.dilations = dilations
        self.dropout_rate = dropout_rate
        self.error_window_length = error_window_length
        self.filters_conv1d = filters_conv1d
        self.nput = None
        self.kernel_size = kernel_size
        self.latent_shape = None
        self.latent_sample_rate = latent_sample_rate
        self.learning_rate = learning_rate
        self.loss = loss
        self.nb_filters = nb_filters
        self.nb_stacks = nb_stacks
        self.padding = padding
        self.pooler = pooler
        self.ts_len = ts_len
        self.ts_dimension = ts_dimension
        self.use_early_stopping = use_early_stopping
            
        # build the model
        self.model = self.build_model()

    def build_model(self, verbose=1):
        K.clear_session()
        encoder = self.build_encoder(summary=False)
        decoder = self.build_decoder(summary=False)
        dec_out = decoder(encoder(self.nput))
        autoencoder = Model([self.nput], [dec_out], name='TCN_Autoencoder')

        adam = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
        autoencoder.compile(loss=self.loss, optimizer=adam, metrics=[self.loss])     
        autoencoder.summary()
        #plot_model(autoencoder, to_file='./tcn_timegan/decoder.png')

        return autoencoder

    def build_encoder(self, verbose = 1, summary=True):
        """Builds the TCN-encoder model.

        If the argument `verbose` isn't passed in, the default verbosity level is used.

        Parameters
        ----------
        verbose : str, optional
            The verbosity level (default is 1)            
        """
        
        sampling_factor = self.latent_sample_rate
        enc_in = self.nput = Input(batch_shape=(None, self.ts_len, self.ts_dimension), name='encoder_input')     
        
        # Put signal through TCN. Output-shape: (batch,sequence length, nb_filters)
        tcn_enc = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
                      padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
                      kernel_initializer=self.conv_kernel_init, name='tcn-enc')(enc_in)
        
        # Now, adjust the number of channels...
        enc_flat = Conv1D(filters=self.filters_conv1d, kernel_size=1, activation=self.activation_conv1d, padding=self.padding)(tcn_enc)

        ## Do some average (max) pooling to get a compressed representation of the time series (e.g. a sequence of length 8)
        enc_pooled = self.pooler(pool_size=sampling_factor, strides=None, padding='valid', data_format='channels_last')(enc_flat)
        
        # If you want, maybe put the pooled values through a non-linear Activation
        latent = Activation("linear")(enc_pooled)
        self.latent_shape = K.int_shape(latent)        
        encoder = Model(inputs=[self.nput], outputs=[latent], name='TCN_Encoder')        
        
        #adam = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
        #encoder.compile(loss=self.loss, optimizer=adam, metrics=[self.loss])
        if summary:     
            encoder.summary()   
            #plot_model(encoder, to_file='./tcn_timegan/encoder.png')     
        
        return encoder

    def build_decoder(self, verbose = 1, summary=True):
        """Builds the TCN-encoder model.

        If the argument `verbose` isn't passed in, the default verbosity level is used.

        Parameters
        ----------
        verbose : str, optional
            The verbosity level (default is 1)            
        """

        sampling_factor = self.latent_sample_rate
        enc_out = Input(batch_shape=self.latent_shape, name='encoder_output')

        # Now we should have a short sequence, which we will upsample again and then try to reconstruct the original series
        dec_upsample = UpSampling1D(size=sampling_factor)(enc_out)

        dec_reconstructed = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
                                padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
                                kernel_initializer=self.conv_kernel_init, name='tcn-dec')(dec_upsample)

        # Put the filter-outputs through a dense layer finally, to get the reconstructed signal
        o = Dense(self.ts_dimension, activation='linear')(dec_reconstructed)
        decoder = Model(inputs=[enc_out], outputs=[o], name='TCN_Decoder')

        #adam = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
        #decoder.compile(loss=self.loss, optimizer=adam, metrics=[self.loss])
        if summary:
            decoder.summary()
            #plot_model(decoder, to_file='./tcn_timegan/decoder.png')
        
        return decoder

    def fit(self, train_X, train_Y, batch_size=32, epochs=40, verbose = 1):
        my_callbacks = None
        if self.use_early_stopping:
            my_callbacks = [EarlyStopping(monitor='val_loss', patience=2, min_delta=1e-4, restore_best_weights=True)]

        keras_verbose = 0
        if verbose > 0:
            print("> Starting the Training...")
            keras_verbose = 2
        start = time.time()
        history = self.model.fit(train_X, train_Y, 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            validation_split=0.001, 
                            shuffle=True,
                            callbacks=my_callbacks,
                            verbose=keras_verbose)
        if verbose > 0:
            print("> Training Time :", round(time.time() - start), "seconds.")

    def predict(self, test_X):
        X_rec =  self.model.predict(test_X)
        return X_rec