import numpy as np
import tensorflow as tf

def random_real_iters(seq_len, n_seq, batch_size):
    def make_random_data():
        while True:
            yield np.random.uniform(low=0, high=1, size=(seq_len, n_seq))
    
    # We use the Python generator to feed a `tf.data.Dataset` that continues to call the random number generator
    # as long as necessary and produces the desired batch size.
    
    random_series = iter(tf.data.Dataset
                         .from_generator(make_random_data, output_types=tf.float32)
                         .batch(batch_size)
                         .repeat())
                         
    real_series = (tf.data.Dataset
                   .from_tensor_slices(data)
                   .shuffle(buffer_size=n_windows)
                   .batch(batch_size))
    real_series_iter = iter(real_series.repeat())
    
    return (random_series, real_series_iter)                 