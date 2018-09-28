"""
@author: Kostas Hatalis
"""
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pylab as plt
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from processing.load_data import load_data
from features.lagged_load import lagged_load
from processing.split_data import split_data
from model.detector import detector
from evaluation.evaluate_results import evaluate_results
from evaluation.output_results import output_results
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#--------------------------------------------------------------------------
experiment = {}
experiment['filename'] = 'data/city.csv'
experiment['lags'] = 48
# tau = np.arange(0.01, 1.0, 0.01)
tau = [0.025, 0.975]
#--------------------------------------------------------------------------
experiment = load_data(experiment)
experiment = lagged_load(experiment)
experiment = split_data(experiment, scaling=MinMaxScaler(), test_size=2/30) # StandardScaler()
#--------------------------------------------------------------------------
experiment['optimizer'] = 'Adam' # SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
experiment['activation'] = 'relu' # relu, sigmoid, tanh, softplus, elu, softsign, sigmoid, linear
experiment['smooth_loss'] = 0 # 0 = pinball, 1 = smooth pinball loss
experiment['maxIter'] = 2000
experiment['batch_size'] = 200
experiment['hidden_dims'] = [40] # number of nodes per hidden layer
experiment['alpha'] = 0.01 # smoothing rate
experiment['Lambda'] = 0.01 # regularization term
experiment['n_tau'] = len(tau)
experiment['tau'] = np.array(tau)
experiment['kappa'] = 1000 # penalty term
experiment['margin'] = 0 # penalty margin
experiment['print_cost'] = 0 # 1 = plot quantile predictions
experiment['plot_results'] = 1 # 1 = plot cost
#--------------------------------------------------------------------------
start_time = time.time()
experiment = detector(experiment,test_method=0) # 0 = QARNET, 1 = QAR
experiment = evaluate_results(experiment)
output_results(experiment)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()