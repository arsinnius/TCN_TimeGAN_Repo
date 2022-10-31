"""
discriminator
embedder
recovery
tcn_generator
tcn_supervisor
"""

from add_fns import make_rnn
from params import (hidden_dim, latent_dim,
                    n_seq, seq_len,
                    n_layers_disc, n_layers_supv,
                    latent_sample_rate, pooler)
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tcn_ae import TCN_AE

# tcn_generator
tcn_ae_gen = TCN_AE(ts_len=seq_len, ts_dimension=n_seq, pooler=pooler)
tcn_generator = tcn_ae_gen.build_encoder()

# supervisor
supervisor = make_rnn(n_layers=n_layers_supv, 
                      input_dim=seq_len//latent_sample_rate,
                      hidden_units=latent_dim, 
                      output_units=latent_dim, 
                      name='Supervisor')
                      
# discriminator
discriminator = make_rnn(n_layers=n_layers_disc, 
                         input_dim=seq_len//latent_sample_rate,                        
                         hidden_units=hidden_dim, 
                         output_units=1, 
                         name='Discriminator')
                         
# embedder and recovery
tcn_ae_embed = TCN_AE(ts_len=seq_len, ts_dimension=n_seq, pooler=pooler)
embedder = tcn_ae_embed.build_encoder()
recovery = tcn_ae_embed.build_decoder() 