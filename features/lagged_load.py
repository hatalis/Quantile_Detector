import pandas as pd

def lagged_load(experiment):
    """
    Computes lagged volumes based on pre-processed data.

    Arguments:
        experiment(dict): Dictionary containing processed volume data (processed_data)
        n(int): Lag-time

    Returns:
        experiment(dict): Experiment dictionary with additional or updated key: X
    """
    Load = experiment['load_data']['Load']
    n = experiment['lags']

    # extract raw price data
    # Load = df['Load']

    # create feature columns
    data, columns = {}, []
    if n > 0:
        for i in range(1, n+1):
            name = 'load_lag_'+str(i)
            data[name] = Load.shift(periods=i)
            columns.append(name)
    else:
        print('Error: n must be greater then 0.')

    feature = pd.DataFrame(data)

    try:
        X = experiment['X']
        X = X.join(feature, how='outer')
    except KeyError:
        X = feature

    experiment['X'] = X

    return experiment