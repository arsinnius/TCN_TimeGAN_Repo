from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam

experiment = 9

## Data Parameters
data_file = 'Qtrly_PDSI_Dataset.csv'
ts_names = ['PDSI', 'PHDI', 'PMDI', 'PZI']
start_date = '1976Q1'
end_date = '2020Q4'

## Training Parameters
n_seq = 4
seq_len = 180
batch_size = 128
train_steps = 10000
gamma = 1
rho = 1

##  Network Parameters
hidden_dim = 4 * n_seq
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
pooler = MaxPooling1D
latent_sample_rate = 4