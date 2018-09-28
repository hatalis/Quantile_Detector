
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(experiment):

    filename = experiment['filename']
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y %H:%M')
    raw_data = pd.read_csv(filename, parse_dates=[0], index_col=0, date_parser=dateparse)
    kWh = raw_data.resample('H').apply('sum')/60 # convert data from kW to kWh (average the power within 1 hour)
    experiment['load_data'] = kWh
    experiment['y'] = kWh
    # plt.plot(kWh)
    # plt.show()

    return experiment
