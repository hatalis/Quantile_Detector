# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 22:55:16 2017

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np

def output_results(experiment):

    SHARP = experiment['SHARP']
    QS = experiment['QS']
    IS = experiment['IS']
    ACE = experiment['ACE']
    CROSS = experiment['CROSS']
    PINC = experiment['PINC']
    plot_results = experiment['plot_results']
    print_cost = experiment['print_cost']
    costs = experiment['costs']
    q_hat = experiment['q_hat']
    y_test = experiment['y_test']
    n_test = experiment['n_test']
    n_tau = experiment['n_tau']

    # print(CROSS)
    # print(QS)
    # print(IS)
    # print(ACE)

    print(PINC)
    print(SHARP)
    print(ACE)
    print(QS)
    print(IS)

    plt.plot(PINC,SHARP)
    if print_cost:
        plt.figure(0)
        plt.plot(np.squeeze(costs))
        plt.ylabel('loss')
        plt.xlabel('epcoh')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.yscale('log')

    if plot_results:
        plt.figure(1)
        plt.plot(y_test, 'r*')
        x = list(range(n_test))
        n_PIs = n_tau // 2
        for i in range(n_PIs):
            y1 = q_hat.T[:,i]
            y2 = q_hat.T[:,-1-i]
            plt.fill_between(x, y1, y2, color='blue', alpha=0.4) # alpha=str(1/n_PIs)
        plt.ylabel('Normalized Load')
        plt.xlabel('Time (hour)')
    return None