from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def split_data(experiment, scaling=None, test_size=0.3, bootstrap=False):
    """
    Splits data into train- and test-set and (optionally) applies scaling to all features.

    Args:

    """
    # load in data from dictionary
    X = experiment['X']
    y = experiment['y']
    experiment['test_size'] = test_size
    experiment['scaling'] = scaling
    experiment['bootstrap'] = bootstrap

    # combine covariates and labels, drop rows with NaN values; do the same thing for returns
    Xy = X.join(y, how='outer')
    Xy = Xy.dropna()

    # redefine X and y from Xy with remaining dates
    X = Xy.drop(columns=['Load'])
    y = Xy[['Load']]

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    split_index = X_test.index.values[0]  # first index of test set (important for evaluation)

    # convert to numpy arrays
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    # randomly resample training data with replacement for bootstrap analysis
    if bootstrap:
        X_train, y_train = resample(X_train, y_train)

    # apply scaling and convert scaled data back to pandas
    if scaling is not None:
        X_train = scaling.fit_transform(X_train)
        X_test = scaling.transform(X_test)
        y_train = scaling.fit_transform(y_train)
        y_test = scaling.transform(y_test)

    # save everything to dictionary
    experiment['X_train'] = X_train
    experiment['y_train'] = y_train
    experiment['n_train'] = len(y_train)
    experiment['X_test'] = X_test
    experiment['y_test'] = y_test
    experiment['n_test'] = len(y_test)
    experiment['split_index'] = split_index

    return experiment
