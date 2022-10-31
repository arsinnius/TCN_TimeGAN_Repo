"""
get_discriminator_loss
get_generator_moment_loss
make_rnn
train_autoencoder_init
train_discriminator
train_embedder
train_generator
train_supervisor
"""

from models import tcn_generator, supervisor
from params import (bce, mse,
                    autoencoder_optimizer, discriminator_optimizer,
                    embedding_optimizer, generator_optimizer, 
                    supervisor_optimizer,
                    gamma)
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (GRU, Dense, 
                                     RNN, Input)                                      
from tqdm import tqdm

import tensorflow as tf

# Autoencoder Training Step
@tf.function
def train_autoencoder_init( x,  
                            auto, encoder, 
                            encoder_loss, decoder, 
                            optimizer=autoencoder_optimizer):
    """
    x:  real data
    auto:   autoencoder
    encoder: embedder model
    encoder_loss: mean squared error
    decoder: recovery model
    optimizer:  autoencoder_optimizer
    
    """
    with tf.GradientTape() as tape:
        x_tilde = auto(x)        
        embedding_loss_t0 = encoder_loss(x, x_tilde)
        e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)
        
    var_list = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(e_loss_0, var_list)
    optimizer.apply_gradients(zip(gradients, var_list))
    return tf.sqrt(embedding_loss_t0)
 
    
## Supervisor Training Step
@tf.function
def train_supervisor(x, encoder, 
                    g_loss, supv, 
                    optimizer=supervisor_optimizer):
    """
    x:  real data
    encoder:    embedder model
    g_loss: mean squared error
    supv:   supervisor model
    optimizer:  supervisor_optimizer
    """
    with tf.GradientTape() as tape:
        h = encoder(x)
        h_hat_supervised = supv(h)
        g_loss_s = g_loss(h[:, 1:, :], h_hat_supervised[:, :-1, :])
    var_list = supv.trainable_variables
    gradients = tape.gradient(g_loss_s, var_list)
    optimizer.apply_gradients(zip(gradients, var_list))
    return g_loss_s


## Mean & Variance Loss
def get_generator_moment_loss(y_true, y_pred):
    y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
    y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
    g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
    g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
    return g_loss_mean + g_loss_var  


## Generator Train Step

#@tf.function
def train_generator(x, z, 
                    adv_supv, adv_emb, encoder, 
                    g_loss_supv, g_loss_unsupv,
                    g_mom_loss, syn_data, gen, supv,
                    optimizer=generator_optimizer):
    """
    x:  real data
    z:  random data
    adv_supv:   adversarial_supervised model
    adv_emb:    adversarial_emb model
    encoder:  embedder model
    g_loss_supv:  mean square error
    g_loss_unsupv:    binary cross-entropy
    g_mom_loss:   get_generator_moment_loss
    syn_data:   synthetic_data model
    gen:    tcn_generator model
    supv:   supervisor model
    optimizer:  generator_optimizer
    """    
    with tf.GradientTape() as tape:
        y_fake = adv_supv(z)
        generator_loss_unsupervised = g_loss_unsupv(y_true=tf.ones_like(y_fake),
                                          y_pred=y_fake)

        y_fake_e = adv_emb(z)
        generator_loss_unsupervised_e = g_loss_unsupv(y_true=tf.ones_like(y_fake_e),
                                            y_pred=y_fake_e)
        h = encoder(x)
        h_hat_supervised = supv(h)
        generator_loss_supervised = g_loss_supv(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        x_hat = syn_data(z)
        generator_moment_loss = g_mom_loss(x, x_hat)

        generator_loss = (generator_loss_unsupervised +
                          generator_loss_unsupervised_e +
                          100 * tf.sqrt(generator_loss_supervised) +
                          100 * generator_moment_loss)

    var_list = gen.trainable_variables + supv.trainable_variables
    gradients = tape.gradient(generator_loss, var_list)
    optimizer.apply_gradients(zip(gradients, var_list))
    return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss  
 
    
## Embedding Train Step

#@tf.function
def train_embedder( x,
                    auto, encoder, decoder, 
                    g_loss_supv, supv, 
                    optimizer=embedding_optimizer):
    """
    x:  real data
    auto:   autoencoder model
    encoder:    embedder model
    decoder:    recovery model
    g_loss_supv:  mean squared error
    supv:   supervisor model
    optimizer:  embedding_optimizer        
    """
    with tf.GradientTape() as tape:
        h = encoder(x)
        h_hat_supervised = supv(h)
        generator_loss_supervised = g_loss_supv(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        x_tilde = auto(x)
        embedding_loss_t0 = g_loss_supv(x, x_tilde)
        e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

    var_list = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(e_loss, var_list)
    optimizer.apply_gradients(zip(gradients, var_list))
    return tf.sqrt(embedding_loss_t0) 

    
## Discriminator Train Step

#@tf.function
def get_discriminator_loss( x, z,
                            adv_emb, adv_supv,
                            d_mod, d_loss):
    """
    x:  real data
    z:  random data
    adv_emb:    adversarial_emb model
    adv_supv:   adversarial_supervised model    
    d_mod:    discriminator_model
    d_loss:    binary cross-entropy
    
    """
    y_real = d_mod(x)
    discriminator_loss_real = d_loss(y_true=tf.ones_like(y_real),
                                  y_pred=y_real)

    y_fake = adv_supv(z)
    discriminator_loss_fake = d_loss(y_true=tf.zeros_like(y_fake),
                                  y_pred=y_fake)

    y_fake_e = adv_emb(z)
    discriminator_loss_fake_e = d_loss(y_true=tf.zeros_like(y_fake_e),
                                    y_pred=y_fake_e)
    return (discriminator_loss_real +
            discriminator_loss_fake +
            gamma * discriminator_loss_fake_e)

#@tf.function
def train_discriminator(x, z,
                        adv_emb, adv_supv,
                        d_mod, d_loss,
                        d_loss_fn, discrim,
                        optimizer=discriminator_optimizer):
    
    """
    x:  real data
    z:  random data
    adv_emb:    adversarial_emb model
    adv_supv:   adversarial_supervised model    
    d_loss:    binary cross-entropy
    d_loss_fn:  get_discriminator_loss function
    d_mod:    discriminator_model    
    discrim:   discriminator
    optimizer:  discriminator optimizer    
    """
    with tf.GradientTape() as tape:
        discriminator_loss = d_loss_fn(x, z,
                                    adv_emb, adv_supv,
                                    d_mod, d_loss)

    var_list = discrim.trainable_variables
    gradients = tape.gradient(discriminator_loss, var_list)
    optimizer.apply_gradients(zip(gradients, var_list))
    return discriminator_loss    

def make_rnn(input_shape, n_layers, hidden_units, output_units, name):
    return Sequential([Input(shape=input_shape)] +      
                      [GRU(units=hidden_units,
                           return_sequences=True,
                           name=f'GRUvy_{i + 1}') for i in range(n_layers)] +
                      [Dense(units=output_units,
                             activation='sigmoid',
                             name='OUT')], name=name)    