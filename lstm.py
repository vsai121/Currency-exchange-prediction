"""
LSTM model
"""

class RNN():
    input_size=1
    num_steps=30
    lstm_size=128
    num_layers=1
    keep_prob=0.8
    batch_size = 64
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    init_epoch = 5
    max_epoch = 50

config = RNN()

import tensorflow as tf
tf.reset_default_graph()
lstm_graph = tf.Graph()    
