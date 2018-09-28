'''
By Kostas Hatalis
'''
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import regularizers
from keras.optimizers import Adam
K.set_floatx('float64')

def detector(experiment,test_method=0):

    # load in training data (x,y)
    X = experiment['X_train']  # data, numpy array of shape (number of features, number of examples)
    y_train = experiment['y_train']  # true observations vector of shape (1, number of examples)
    Xt = experiment['X_test']
    y_test = experiment['y_test']
    smooth_loss = experiment['smooth_loss']
    kappa = experiment['kappa']
    alpha = experiment['alpha']
    margin = experiment['margin']
    Lambda = experiment['Lambda']
    tau = experiment['tau']
    hidden_dims = experiment['hidden_dims']
    maxIter = experiment['maxIter']
    batch_size = experiment['batch_size']
    activation = experiment['activation']
    optimizer = experiment['optimizer']
    lags = experiment['lags']
    n_tau = experiment['n_tau']
    layers_dims = [lags]+hidden_dims+[n_tau]

    # -------------------------------------- build the model
    model = Sequential()
    if test_method == 0: # QARNN
        for i in range(0,len(layers_dims)-2):
            model.add(Dense(layers_dims[i+1], input_dim=layers_dims[i], kernel_regularizer=regularizers.l2(Lambda),
                            kernel_initializer='normal', activation=activation))
        model.add(Dense(layers_dims[-1], kernel_initializer='normal'))
    elif test_method == 1: # QAR
        model.add(Dense(layers_dims[-1], input_dim=layers_dims[0], kernel_regularizer=regularizers.l2(Lambda),
                        kernel_initializer='normal'))

    # -------------------------------------- compile and fit model
    model.compile(loss=lambda Y, Q: pinball_loss(tau,Y,Q,alpha,smooth_loss,kappa,margin), optimizer=optimizer)
    history = model.fit(X, y_train, epochs=maxIter, verbose=0, batch_size=batch_size)

    # -------------------------------------- estimate quantiles of testing data
    Xt[20,0] = 1.5
    y_test[20] = 1.5
    q_hat = model.predict(Xt) #+np.random.uniform(size=np.shape(Xt))

    experiment['y_test'] = y_test
    experiment['q_hat'] = q_hat.T
    experiment['costs'] = history.history['loss']

    return experiment




# pinball loss function with penalty
def pinball_loss(tau, y, q, alpha = 0.01, smooth_loss = 1, kappa=0, margin=0):
    error = (y - q)
    diff = q[:, 1:] - q[:, :-1]

    if smooth_loss == 0: # pinball function
        quantile_loss = K.mean(K.maximum(tau * error, (tau - 1) * error))
    elif smooth_loss == 1: # smooth pinball function
        quantile_loss = K.mean(tau * error + alpha * K.softplus(-error / alpha))
    elif smooth_loss == 2: # huber norm approximation
        epsilon = 2 ** -8

        # if K.abs(error) > epsilon:
        #     u = K.abs(error) - epsilon / 2
        # else:
        #     u = (error**2) / (2 * epsilon)

        logic = K.cast((K.abs(error) > epsilon),dtype='float64')

        u = (K.abs(error)-epsilon/2)*logic + ((error**2) / (2 * epsilon))*(1-logic)

        quantile_loss = K.mean(K.maximum(tau * u, (tau - 1) * u))


    # penalty = -kappa * K.mean(alpha2*K.softplus(-diff / alpha2))
    # penalty = K.mean(K.maximum(tf.Variable(tf.zeros([1], dtype=tf.float64)), margin - diff)) * kappa
    penalty = kappa * K.mean(tf.square(K.maximum(tf.Variable(tf.zeros([1], dtype=tf.float64)), margin - diff)))

    return quantile_loss + penalty


