import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import random


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT:
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    with open(filename, 'r') as file:
        csvreader = csv.reader(file)
        field = next(csvreader)
        for row in csvreader:
            dataset.append([float(num) for num in row[1:]])
    dataset = np.array(dataset)
    return dataset


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on.
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    num_datapoints = len(dataset)
    sample_mean = 0
    sample_std = 0
    n = len(dataset.T[col]) # n value for mean and std calculation

    sum_of_col = 0
    # get sum of values in col for sigma
    for num in dataset.T[col]:
        sum_of_col += num
    sample_mean = sum_of_col / n

    std_sum = 0
    # get sum of (xi - xbar)^2 in col
    for num in dataset.T[col]:
        std_sum += (num - sample_mean)**2
    sample_std = math.sqrt(std_sum / (n - 1))

    print("%d" % num_datapoints)
    print("%.2f" % sample_mean) # ".2f" -> only print to two decimal places
    print("%.2f" % sample_std)
    # pass


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    mse = 0

    n = len(dataset)
    # iterate through all elements in each feature vector x
    for row in dataset:
        squared_error = betas[0] # for B0
        for i in range(len(cols)):
            squared_error += row[cols[i]] * betas[i+1] # for BmXim
        squared_error -= row[0] # for -yi
        squared_error = squared_error**2
        mse += squared_error

    mse /= n
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = [0] * len(betas)
    n = len(dataset)


    # want to append to grads same number of values we have in betas
    for betaNum in range(len(betas)):

        if betaNum == 0:
            # iterate through all elements in each feature vector x
            for row in dataset:
                partial_deriv = betas[0] - row[0] # for -yi
                for i in range(len(cols)):
                    partial_deriv += row[cols[i]] * betas[i+1] # for BmXim
                grads[0] += partial_deriv
            grads[0] *= (2 / n)
        else:
            # iterate through all elements in each feature vector x
            for row in dataset:
                partial_deriv = betas[0] - row[0] # for -yi
                for i in range(len(cols)):
                    partial_deriv += row[cols[i]] * betas[i+1] # for BmXim
                # multiple partial deriv by relevant col value for that row
                grads[betaNum] += (partial_deriv * row[cols[betaNum-1]])
            grads[betaNum] *= (2 / n)
    grads = np.array(grads)
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    newBetas = betas
    for time in range(1,T+1):
        gradient = gradient_descent(dataset, cols, betas)
        for currBeta in range(len(betas)): # update betas
            newBetas[currBeta] -= eta * gradient[currBeta]
        mse = regression(dataset, cols, betas) # get mse
        print("{}".format(time), end=" ")
        print("{:.2f}".format(mse), end=" ")
        print("{:.2f}".format(newBetas[0]), end=" ")
        print(" ".join('{:.2f}'.format(f) for f in newBetas[1:]))


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    betas = None
    mse = None

    # first column needs to be all 1's
    x_matrix = [[1] * len(dataset)]
    # append rest of xi columns
    for col in cols:
        x_matrix.append(dataset[:,col])
    x_matrix = np.array(x_matrix)
    x_matrix = x_matrix.transpose()
    # build our y
    y_matrix = dataset[:,0]
    y_matrix = np.array(y_matrix)
    y_matrix = y_matrix.transpose()

    # build our xT
    x_transposed = x_matrix.transpose()

    result = np.dot(np.dot(np.linalg.inv(np.dot(x_transposed,x_matrix)), x_transposed), y_matrix)
    betas = result.tolist() # need an iterable when using '*' in our return statement
    mse = regression(dataset, cols, result)

    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    # f(x) = B0 + B1X1 + B2X2 + ... + BmXm
    betas = compute_betas(dataset, cols)
    result = betas[1] # B0

    for i in range(2,len(betas)):
        result += betas[i] * features[i-2]

    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    linear_dataset = []
    quadratic_dataset = []

    for point in X:
        z = np.random.normal(0.0, sigma)
        linear_dataset.append(np.array([betas[0] + betas[1]*point[0] + z, point[0]]))
        z = np.random.normal(0.0, sigma)
        quadratic_dataset.append(np.array([alphas[0] + alphas[1]*(point[0]**2) + z, point[0]]))

    return np.array(linear_dataset), np.array(quadratic_dataset)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    # step 1
    input_X = []
    for i in range(1000):
        input_X.append([random.randint(-100,100)])
    input_X = np.array(input_X)
    # step 2
    betas = np.array([1,2])
    alphas = np.array([2,3])
    # step 3
    sigmas = [10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5]

    data_mse = []
    for sigma in sigmas:
        # step 4
        linear_dataset, quadratic_dataset = synthetic_datasets(betas, alphas, input_X, sigma)
        # step 5
        data_mse.append([compute_betas(linear_dataset, [1])[0], compute_betas(quadratic_dataset, [1])[0]])
    # step 6
    fig = plt.figure()
    linear_mse = []
    quadratic_mse = []
    for mse in data_mse:
        linear_mse.append(mse[0])
        quadratic_mse.append(mse[1])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sigma')
    plt.ylabel('MSE')
    plt.plot(sigmas, linear_mse, marker="o", label='linear')
    plt.plot(sigmas, quadratic_mse, marker="o", label='quadratic')
    plt.legend()
    plt.savefig('mse.pdf')

if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
