import tensorflow as tf
from setting import *
from keras.models import Model
from keras.layers import MultiHeadAttention, Input, BatchNormalization, Dot, Conv1D, Attention, LSTM, Softmax, Dense, \
    MaxPool1D

# Prediction model for well group oil production
def myModel(shape1, shape2):
    model_in1 = Input(shape=(shape1[1], shape1[2]))
    model_in2 = Input(shape=(shape2[1], shape2[2]))

    '''IPFP Module'''
    x1 = LSTM(64, return_sequences=True)(model_in1)
    x1 = LSTM(32, return_sequences=True)(x1)
    x1 = LSTM(16, return_sequences=True)(x1)
    x1 = LSTM(4, return_sequences=True)(x1)
    x1 = Attention()([x1, x1])

    x2 = LSTM(64, return_sequences=True)(model_in2)
    x2 = LSTM(32, return_sequences=True)(x2)
    x2 = LSTM(16, return_sequences=True)(x2)
    x2 = LSTM(4, return_sequences=True)(x2)
    x2 = Attention()([x2, x2])
    x2 = tf.transpose(x2, [0, 2, 1])

    '''IPFF Module'''
    x3_1 = Dot(axes=(2, 1))([x1, x2])
    x3_2 = Softmax()(x3_1)
    x3_2 = x3_1 * x3_2
    x3 = BatchNormalization()(x3_2)

    '''MSFE Module'''
    # Multi-seale CNN: First scale
    x3_1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x3)
    x3_1 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(x3_1)
    x3_1 = Conv1D(filters=8, kernel_size=3, padding='same', activation='relu')(x3_1)
    x3_1 = MaxPool1D(pool_size=2, strides=2)(x3_1)

    # Multi-seale CNN: Second scale
    x3_2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(x3)
    x3_2 = Conv1D(filters=16, kernel_size=5, padding='same', activation='relu')(x3_2)
    x3_2 = Conv1D(filters=8, kernel_size=5, padding='same', activation='relu')(x3_2)
    x3_2 = MaxPool1D(pool_size=2, strides=2)(x3_2)

    # Multi-seale CNN: Third scale
    x3_3 = Conv1D(filters=32, kernel_size=7, padding='same', activation='relu')(x3)
    x3_3 = Conv1D(filters=16, kernel_size=7, padding='same', activation='relu')(x3_3)
    x3_3 = Conv1D(filters=8, kernel_size=7, padding='same', activation='relu')(x3_3)
    x3_3 = MaxPool1D(pool_size=2, strides=2)(x3_3)

    # Merge multi-scale features
    x3 = tf.concat([x3_1, x3_2, x3_3], axis=-1)

    x4 = MultiHeadAttention(num_heads=8, key_dim=3)(x3, x3)
    x4 = LSTM(units=16, return_sequences=True)(x4)
    x4 = LSTM(units=8, return_sequences=True)(x4)
    x4 = LSTM(units=4, return_sequences=True)(x4)
    x4 = LSTM(units=1, return_sequences=False)(x4)
    x4 = Dense(1)(x4)
    x4 = tf.reshape(x4, (-1, 1))
    model4_out = x4

    model = Model(inputs=[model_in1, model_in2], outputs=model4_out)
    return model