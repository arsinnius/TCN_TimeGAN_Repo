#%cd '/content/drive/MyDrive/ColabNotebooks/TSGANs/TimeGAN_TF2'
from add_fns import make_random_data
from data_fns import get_fred_data
from models import (embedder, discriminator, 
                    recovery, tcn_generator, supervisor)
from params import (experiment, train_steps, data_name,
                    batch_size, seq_len, synthetic_data_name,
                    hidden_dim, latent_dim, 
                    autoencoder_optimizer, embedding_optimizer,
                    discriminator_optimizer, generator_optimizer,
                    supervisor_optimizer, mse, bce)
                    
from train_fns import (get_discriminator_loss, get_generator_moment_loss, 
                       train_autoencoder_init, train_discriminator,
                       train_embedder, train_generator, train_supervisor)
                       
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
import time

sns.set_style('white')

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

## Experiment Path
results_path = Path('./tcn_timegan')
if not results_path.exists():
    results_path.mkdir()

log_dir = results_path / f'experiment_{experiment:02}'
if not log_dir.exists():
    log_dir.mkdir(parents=True)
    
sample_dir = log_dir / 'samples'
if not sample_dir.exists():
    sample_dir.mkdir(parents=True)

eval_dir = log_dir / 'evals'
if not eval_dir.exists():
    eval_dir.mkdir(parents=True)
     
hdf_store = results_path / 'TimeSeriesGAN.h5'

## Get Data and Plot Series
df = get_fred_data(hdf=hdf_store, 
                  fname=data_name,
                  save_dir=log_dir,
                  plot_series=True,
                  show_corr=True)
n_seq = len(df.columns)                  

## Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df).astype(np.float32)

## Create rolling window sequences
data = []
for i in range(len(df) - seq_len):
    data.append(scaled_data[i:i + seq_len])

n_windows = len(data)

## Create tf.data.Dataset
real_series = (tf.data.Dataset
               .from_tensor_slices(data)
               .shuffle(buffer_size=n_windows)
               .batch(batch_size))
               
real_series_iter = iter(real_series.repeat())

## Set up random series generator
random_series = iter(tf.data.Dataset
                     .from_generator(make_random_data, output_types=tf.float32)
                     .batch(batch_size)
                     .repeat())                                          
                     
## TimeGAN Components

## Set up logger
writer = tf.summary.create_file_writer(log_dir.as_posix())

# TimeGAN Training

## Input place holders
X = Input(shape=[seq_len, n_seq], name='RealData')
Z = Input(shape=[seq_len, n_seq], name='RandomData')

## Model Inputs and Outputs

### Autoencoder
H = embedder(X)
X_tilde = recovery(H)

### Synthetic Data
E_hat = tcn_generator(Z)
H_hat = supervisor(E_hat)
X_hat = recovery(H_hat)

### Discriminator Output
Y_real = discriminator(H)
Y_fake = discriminator(H_hat)
Y_fake_e = discriminator(E_hat)

## Phase 1: Autoencoder Training
autoencoder = Model(inputs=X,
                    outputs=X_tilde,
                    name='Autoencoder')
autoencoder.summary()                    
                    
print(color.BLUE + "\nAUTOENCODER TRAINING LOOP" + color.END)
for step in tqdm(range(train_steps)):
    X_ = next(real_series_iter)
    step_e_loss_t0 = train_autoencoder_init(x=X_, 
                                            auto=autoencoder, 
                                            encoder=embedder, 
                                            encoder_loss=mse,
                                            decoder=recovery)
    with writer.as_default():
        tf.summary.scalar('Loss Autoencoder Init', step_e_loss_t0, step=step)

## Persist model
autoencoder.save(log_dir / 'autoencoder')


# Phase 2: Supervised training

print(color.BLUE + "\nSUPERVISOR TRAINING LOOP" + color.END)
for step in tqdm(range(train_steps)):
    X_ = next(real_series_iter)
    step_g_loss_s = train_supervisor(x=X_, 
                                    encoder=embedder,
                                    g_loss=mse,
                                    supv=supervisor)
    with writer.as_default():
        tf.summary.scalar('Loss Generator Supervised Init', step_g_loss_s, step=step)

## Persist Model
supervisor.save(log_dir / 'supervisor')


# Joint Training

### Adversarial Architecture - Supervised
adversarial_supervised = Model(inputs=Z,
                               outputs=Y_fake,
                               name='AdversarialNetSupervised')
adversarial_supervised.summary()

### Adversarial Architecture in Latent Space
adversarial_emb = Model(inputs=Z,
                    outputs=Y_fake_e,
                    name='AdversarialNet')
adversarial_emb.summary()


## Discriminator

### Architecture: Real Data
discriminator_model = Model(inputs=X,
                            outputs=Y_real,
                            name='DiscriminatorReal')
discriminator_model.summary()


### Synthetic Data
synthetic_data = Model(inputs=Z,
                       outputs=X_hat,
                       name='SyntheticData')
synthetic_data.summary()

print(color.BLUE + "\nJOINT TRAINING LOOP" + color.END)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
for step in range(train_steps):
    # Train generator (twice as often as discriminator)
    for kk in range(2):
        X_ = next(real_series_iter)
        Z_ = next(random_series)

        # Train generator
        step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(x=X_, z=Z_, 
                                                                    adv_supv=adversarial_supervised, 
                                                                    adv_emb=adversarial_emb, 
                                                                    encoder=embedder, 
                                                                    g_loss_supv=mse, 
                                                                    g_loss_unsupv=bce,
                                                                    g_mom_loss=get_generator_moment_loss,
                                                                    syn_data=synthetic_data,
                                                                    gen=tcn_generator,
                                                                    supv=supervisor)
        # Train embedder
        step_e_loss_t0 = train_embedder(x=X_,
                                        auto=autoencoder,
                                        encoder=embedder,
                                        decoder=recovery,
                                        g_loss_supv=mse,
                                        supv=supervisor)
    X_ = next(real_series_iter)
    Z_ = next(random_series)
    step_d_loss = get_discriminator_loss(x=X_, 
                                        z=Z_,
                                        adv_emb=adversarial_emb,
                                        adv_supv=adversarial_supervised,
                                        d_mod=discriminator_model, 
                                        d_loss=bce)
    if step_d_loss > 0.15:
        step_d_loss = train_discriminator(x=X_, 
                                        z=Z_,
                                        adv_emb=adversarial_emb,
                                        adv_supv=adversarial_supervised,
                                        d_loss=bce,
                                        d_loss_fn=get_discriminator_loss,
                                        d_mod=discriminator_model,
                                        discrim=discriminator)
    if step % 100 == 0:
        print(f'{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
              f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')

    with writer.as_default():
        tf.summary.scalar('G Loss S', step_g_loss_s, step=step)
        tf.summary.scalar('G Loss U', step_g_loss_u, step=step)
        tf.summary.scalar('G Loss V', step_g_loss_v, step=step)
        tf.summary.scalar('E Loss T0', step_e_loss_t0, step=step)
        tf.summary.scalar('D Loss', step_d_loss, step=step)

synthetic_data.save(log_dir / synthetic_data_name) 

## Plot Models
"""
plot_model(adversarial_emb, show_shapes=True)
plot_model(adversarial_supervised, show_shapes=True)
plot_model(discriminator, show_shapes=True) 
plot_model(discriminator_model, show_shapes=True)
plot_model(synthetic_data, show_shapes=True)
plot_model(tcn_generator, show_shapes=True)
plot_model(supervisor, show_shapes=True)
"""
