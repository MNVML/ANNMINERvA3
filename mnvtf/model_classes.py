import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization


LOGGER = logging.getLogger(__name__)


class ConvModel(keras.Model):

    def __init__(self, num_classes=6):
        super(ConvModel, self).__init__(name='conv_model')
        self.num_classes = num_classes
        # x branch
        self.x_conv_1 = Conv2D(8, kernel_size=(6, 5), activation='relu')
        self.x_conv_2 = Conv2D(12, kernel_size=(5, 5), activation='relu')
        self.x_conv_3 = Conv2D(16, kernel_size=(5, 5), activation='relu')
        self.x_max_pool1 = MaxPooling2D(pool_size=(2, 2))
        self.x_dense1 = Dense(48, activation='relu')
        self.x_dropout1 = Dropout(0.25)
        # u branch
        self.u_conv_1 = Conv2D(8, kernel_size=(6, 5), activation='relu')
        self.u_conv_2 = Conv2D(12, kernel_size=(5, 5), activation='relu')
        self.u_conv_3 = Conv2D(16, kernel_size=(5, 5), activation='relu')
        self.u_max_pool1 = MaxPooling2D(pool_size=(2, 2))
        self.u_dense1 = Dense(48, activation='relu')
        self.u_dropout1 = Dropout(0.25)
        # v branch
        self.v_conv_1 = Conv2D(8, kernel_size=(6, 5), activation='relu')
        self.v_conv_2 = Conv2D(12, kernel_size=(5, 5), activation='relu')
        self.v_conv_3 = Conv2D(16, kernel_size=(5, 5), activation='relu')
        self.v_max_pool1 = MaxPooling2D(pool_size=(2, 2))
        self.v_dense1 = Dense(48, activation='relu')
        self.v_dropout1 = Dropout(0.25)
        # combined
        self.flatten = Flatten()
        self.dense_1 = Dense(96, activation='relu')
        self.dropout2 = Dropout(0.5)
        self.dense_2 = Dense(num_classes)
        LOGGER.info('initialized ConvModel')

    def call(self, x_inputs, u_inputs, v_inputs):
        # define forward pass using layers defined in `__init__`
        # use `BatchNormalization` for now, but it looks weird in the tboard
        # graph...
        # x
        x = self.x_conv_1(x_inputs)
        x = BatchNormalization()(x)
        x = self.x_conv_2(x)
        x = BatchNormalization()(x)
        x = self.x_conv_3(x)
        x = BatchNormalization()(x)
        x = self.x_max_pool1(x)
        x = self.x_dropout1(x)
        x = self.flatten(x)
        x = self.x_dense1(x)
        x = BatchNormalization()(x)
        # u
        u = self.u_conv_1(u_inputs)
        u = BatchNormalization()(u)
        u = self.u_conv_2(u)
        u = BatchNormalization()(u)
        u = self.u_conv_3(u)
        u = BatchNormalization()(u)
        u = self.u_max_pool1(u)
        u = self.u_dropout1(u)
        u = self.flatten(u)
        u = self.u_dense1(u)
        u = BatchNormalization()(u)
        # v
        v = self.v_conv_1(v_inputs)
        v = BatchNormalization()(v)
        v = self.v_conv_2(v)
        v = BatchNormalization()(v)
        v = self.v_conv_3(v)
        v = BatchNormalization()(v)
        v = self.v_max_pool1(v)
        v = self.v_dropout1(v)
        v = self.flatten(v)
        v = self.v_dense1(v)
        v = BatchNormalization()(v)
        conc = concatenate([x, u, v], axis=1)
        m = self.dense_1(conc)
        m = BatchNormalization()(m)
        m = self.dropout2(m)
        return self.dense_2(m)

    def compute_output_shape(self, input_shape):
        # we must override this function if we want to use the subclass
        # model as part of a functional-style model - otherwise it is
        # optional
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
