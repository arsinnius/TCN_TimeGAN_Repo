from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam

## Settings
n_seq = 10
data_name = '/data/stress_vars_df'
synthetic_data_name = 'synthetic_stress_vars_df'
experiment = 8
train_steps = 10000
gamma = 1

## Parameters
seq_len = 24
batch_size = 128

##  Network Parameters
hidden_dim = 24
latent_dim = 8
n_layers_disc = 3
n_layers_supv = 2

## Generic Loss Functions
mse = MeanSquaredError()
bce = BinaryCrossentropy()

## Optimizers
autoencoder_optimizer = Adam()
embedding_optimizer = Adam()
discriminator_optimizer = Adam()
generator_optimizer = Adam()
supervisor_optimizer = Adam()

## Pooling Parameters
pooler = AveragePooling1D
latent_sample_rate = 6