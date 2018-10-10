import keras.backend as K
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import time

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge
from keras.layers import Convolution2D, Conv2DTranspose, BatchNormalization, MaxPooling2D
from keras.layers.convolutional import _Conv
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import Concatenate
from keras.legacy import interfaces
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.engine import InputSpec
from keras import backend as K
from keras import objectives
from util import (disc_mutual_info_loss, sample_unit_gaussian,
                  sample_categorical, plot_digit_grid, EPSILON)

class InfoGAN():
    """
    Class to handle building and training InfoGAN models.
    """
    def __init__(self, input_shape=(28, 28, 1), latent_dim=62, reg_cont_dim=0,
                 reg_disc_dim=10, filters=(64, 64, 64),
                 aux_filters=(64, 64, 64), batch_size=128):
        """
        Setting up everything.

        Parameters
        ----------
        input_shape : Array-like, shape (num_rows, num_cols, num_channels)
            Shape of image.

        latent_dim : int
            Dimension of latent distribution.

        reg_cont_dim : int
            Dimension of continuous latent regularized distribution.

        reg_disc_dim : int
            Dimension of discrete latent regularized distribution.

        filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution in increasing order of depth.

        aux_filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution for auxiliary distribution
            in increasing order of depth.
        """
        self.generator = None
        self.discriminator = None
        self.batch_size = batch_size
        self.callbacks = []
        self.opt = None
        self.gan = None
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.reg_disc_dim = reg_disc_dim
        self.reg_cont_dim = reg_cont_dim
        self.total_latent_dim = latent_dim + reg_disc_dim + reg_cont_dim
        self._setup_model()

    def _setup_model(self):
        """
        Method to set up model.
        """
        self._setup_generator()
        self._setup_discriminator()
        self._setup_auxiliary()
        self._setup_gan()

    def _setup_generator(self):
        """
        Set up generator G
        """
        self.z_input = Input(batch_shape=(self.batch_size, self.latent_dim), name='z_input')
        self.c_disc_input = Input(batch_shape=(self.batch_size, self.reg_disc_dim), name='c_input')
        x = Concatenate()([self.z_input, self.c_disc_input])
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(16 * 16 * 64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Reshape((16, 16, 64))(x)
        x = Conv2DTranspose(64, (4, 4), strides=(2,2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(32, (4, 4), strides=(2,2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(16, (4, 4), strides=(2,2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(8, (4, 4), strides=(2,2), padding='same', activation='relu')(x)
        self.g_output = Conv2DTranspose(3, (4, 4), strides=(2,2), padding='same',
                                        activation='sigmoid', name='generated')(x)

        self.generator = Model(inputs=[self.z_input, self.c_disc_input], outputs=[self.g_output], name='gen_model')
        self.opt_generator = Adam(lr=1e-3)
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=self.opt_generator)

    def _setup_discriminator(self):
        """
        Set up discriminator D
        """
        self.d_input = Input(batch_shape=(self.batch_size,) + self.input_shape, name='d_input')
        x = SNConv2D(64,(3,3), strides=(2,2), padding="same")(self.d_input)
        x = LeakyReLU(0.2)(x)
        x = SNConv2D(64,(3,3), strides=(2,2), padding="same")(x)
        x = LeakyReLU(0.2)(x)
        x = SNConv2D(64,(3,3), strides=(2,2), padding="same")(x)
        x = LeakyReLU(0.2)(x)
        x = SNConv2D(128,(3,3), strides=(2,2), padding="same")(x)
        x = LeakyReLU(0.2)(x)
        x = SNConv2D(256,(3,3), strides=(2,2), padding="same")(x)
        x = LeakyReLU(0.2)(x)
        x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(0.1)(x)

        self.d_hidden = BatchNormalization()(x) # Store this to set up Q
        self.d_output = Dense(1, activation='sigmoid', name='d_output')(self.d_hidden)

        self.discriminator = Model(inputs=[self.d_input], outputs=[self.d_output], name='dis_model')
        self.opt_discriminator = Adam(lr=2e-4)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.opt_discriminator)

    def _setup_auxiliary(self):
        """
        Setup auxiliary distribution.
        """
        x = Dense(128)(self.d_hidden)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        self.q_output = Dense(self.reg_disc_dim, activation='softmax', name='auxiliary')(x)
        self.auxiliary = Model(inputs=[self.d_input], outputs=[self.q_output], name='aux_model')
        # It does not matter what the loss is here, as we do not specifically train this model
        self.auxiliary.compile(loss='mse', optimizer=self.opt_discriminator)

    def _setup_gan(self):
        """
        Set up GAN
        Discriminator weights should not be trained with the GAN.
        """
        self.discriminator.trainable = False
        import pdb;pdb.set_trace()
        gan_output = self.discriminator(self.g_output)
        gan_output_aux = self.auxiliary(self.g_output)
        self.gan = Model(inputs=[self.z_input, self.c_disc_input], outputs=[gan_output, gan_output_aux])
        self.gan.compile(loss={'dis_model' : 'binary_crossentropy',
                               'aux_model' : disc_mutual_info_loss},
                         loss_weights={'dis_model' : 1.,
                                       'aux_model' : -1.},
                         optimizer=self.opt_generator)

    def sample_latent_distribution(self):
        """
        Returns continuous and discrete samples from latent distribution
        """
        z = sample_unit_gaussian(self.batch_size, self.latent_dim)
        c_disc = sample_categorical(self.batch_size, self.reg_disc_dim)
        return z, c_disc

    def generate(self):
        """
        Generate a batch of examples.
        """
        z, c_disc = self.sample_latent_distribution()
        return self.generator.predict([z, c_disc], batch_size=self.batch_size)

    def discriminate(self, x_batch):
        """
        """
        return self.discriminator.predict(x_batch, batch_size=self.batch_size)

    def get_aux_dist(self, x_batch):
        """
        """
        return self.auxiliary.predict(x_batch, batch_size=self.batch_size)

    def plot(self):
        """
        """
        return plot_digit_grid(self)

class SNConv2D(_Conv):
    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(SNConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.input_spec = InputSpec(ndim=4)
        self.Ip = 1
        self.u = self.add_weight(
            name='W_u',
            shape=(1,filters),
            initializer='random_uniform',
            trainable=False
        )

    def call(self, inputs):
        outputs = K.conv2d(
            inputs,
            self.W_bar(),
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


    def get_config(self):
        config = super(SNConv2D, self).get_config()
        config.pop('rank')
        return config

    def W_bar(self):
        # Spectrally Normalized Weight
        W_mat = K.permute_dimensions(self.kernel, (3, 2, 0, 1)) # (h, w, i, o) => (o, i, h, w)
        W_mat = K.reshape(W_mat,[K.shape(W_mat)[0], -1]) # (o, i * h * w)

        if not self.Ip >= 1:
            raise ValueError("The number of power iterations should be positive integer")

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = _l2normalize(K.dot(_u, W_mat))
            _u = _l2normalize(K.dot(_v, K.transpose(W_mat)))

        sigma = K.sum(K.dot(_u,W_mat)*_v)

        K.update(self.u,K.in_train_phase(_u, self.u))
        return self.kernel / sigma

def _l2normalize(x):
    return x / K.sqrt(K.sum(K.square(x)) + K.epsilon())
