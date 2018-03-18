import tensorflow as tf
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import config

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)


def init():
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights(config.KERAS_NIMA_MODEL_PATH)

    optimizer = Adam(lr=1e-3)
    model.compile(optimizer, loss=earth_mover_loss)

    model.compile(optimizer, loss=earth_mover_loss)
    model.summary()
    graph = tf.get_default_graph()

    return model, graph