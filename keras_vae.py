import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras import optimizers

import mnist_data as MD

class VAE:
    def __init__(self,**kwargs):
        self.original_dim = kwargs.get('original_dim',(28,28))
        self.npixels = np.prod(self.original_dim)
        self.dim_per_view = kwargs.get('dim_per_view',3)
        self.views = kwargs.get('views',12)
        self.latent_dim = self.dim_per_view * self.views
        self.intermediate_dim = kwargs.get('intermediate_dim',512)
        self.epsilon_std = kwargs.get('epsilon_std',1.0)
        self.beta = kwargs.get('beta',0.1)
        self.optimizer = kwargs.get('optimizer',None)
        if not self.optimizer:
            self.optimizer = optimizers.Adam()
        self.__init()

    def __init(self):
        x = Input(shape=self.original_dim)
        h = self.encunit(x,n_hidden=1)
        z, kl = self.sampling(h)

        # we instantiate these layers separately so as to reuse them later
        x_decoded_mean, declayers = self.decunit(z,n_hidden=1,layers=None)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = self.npixels * metrics.binary_crossentropy(x, x_decoded_mean)
            return K.mean(xent_loss + self.beta*kl)

        self.vae = Model(x, x_decoded_mean)
        self.vae.compile(optimizer=self.optimizer, loss=vae_loss)

        # build a model to project inputs on the latent space
        self.encoder = Model(x, z)
        # build a digit generator that can sample from the learned distribution
        z4d = Input(shape=(self.latent_dim,))
        _x_decoded_mean = self.decunit(z4d,n_hidden=1,layers=declayers)
        self.decoder = Model(z4d, _x_decoded_mean)

    def sampling(self,h):
        z_mean = Dense(self.latent_dim, name='z_mean')(h)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(h)

        def _sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                                      stddev=self.epsilon_std)
            z_sampled = z_mean + K.exp(z_log_var / 2) * epsilon
            return K.in_train_phase(z_sampled,z_mean)

        z = Lambda(_sampling, output_shape=(self.latent_dim,),name='sampling')([z_mean, z_log_var])
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        return z, kl_loss

    def encunit(self,x,n_hidden=1):
        h = x
        h = Flatten()(h)
        for i in range(n_hidden):
            h = Dense(self.intermediate_dim, activation='relu', name='enc_dense_%i'%(i))(h)
        return h

    def decunit(self,z,n_hidden=1,layers=None):
        xd = z
        if not layers:
            layers = []
            for i in range(n_hidden):
                layers.append(Dense(self.intermediate_dim, activation='relu', name='dec_dense_%i'%(i)))
            layers.append(Dense(self.npixels, activation='sigmoid', name='dec_dense_last'))
            layers.append(Reshape(self.original_dim))
            for layer in layers:
                xd = layer(xd)
            return xd,layers
        else:
            for layer in layers:
                xd = layer(xd)
            return xd

if __name__ == "__main__":
    batch_size = 100
    epochs = 10
    # train the VAE on MNIST digits
    d = MD.MNIST()
    vaemodel = VAE()
    vaemodel.vae.fit_generator(d.generator(batch_size=batch_size),epochs=epochs,steps_per_epoch=60000/batch_size)
