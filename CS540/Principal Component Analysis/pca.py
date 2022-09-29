from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # TODO: add your code here
    originalData = np.load(filename)
    meanArr = np.mean(originalData, axis=0)
    centeredArr = originalData - meanArr
    return centeredArr

def get_covariance(dataset):
    # TODO: add your code here
    n = len(dataset)
    datasetTransposed = np.transpose(dataset)
    dotProduct = np.dot(datasetTransposed,dataset)
    covariance = dotProduct / (n-1)
    return covariance

def get_eig(S, m):
    # TODO: add your code here
    eigenvals, eigenvectors = eigh(S, eigvals=(len(S)-m,len(S)-1))
    eigenvals = eigenvals[::-1] # to make descending/reverse it
    eigenvals = np.diag(eigenvals) # yields a diagonal matrix
    eigenvectors = eigenvectors[:,::-1]
    return eigenvals, eigenvectors

def get_eig_perc(S, perc):
    # TODO: add your code here
    eigenvals, eigenvectors = eigh(S)
    eigenvals = eigenvals[::-1] # to make descending/reverse it
    eigenvectors = eigenvectors[:,::-1]
    eigenvalsSum = np.sum(eigenvals)
    eigenvalsPerc = []
    eigenvectorsPerc = eigenvectors
    invalidCols = []
    for i in range(len(eigenvals)):
        # compute percentage of variance
        eigenPercent = eigenvals[i] / eigenvalsSum
        if eigenPercent > perc:
            eigenvalsPerc.append(eigenvals[i])
            # eigenvectorsPerc.append(eigenvectors[i])
        else:
            # eigenvectorsPerc = np.delete(eigenvectorsPerc, i, 1)
            invalidCols.append(i)

    # axis=1 indicates deleting a column
    eigenvectorsPerc = np.delete(eigenvectors, invalidCols, 1)

    # change eigenvalsPerc back into a numpy array and then a diagonal matrix
    eigenvalsPerc = np.diag(np.array(eigenvalsPerc))

    return eigenvalsPerc, eigenvectorsPerc

def project_image(img, U):
    #print("img shape",img.shape)
    #print("U shape",U.shape)
    #alpha = np.dot(np.transpose(U), img)
    alpha = np.dot(img, U)
    #print("alpha shape",alpha.shape)
    projection = np.dot(U, alpha)
    return projection

def display_image(orig, proj):
    # reshaping the images to be 32x32
    original = np.reshape(orig,(32,32), order='F')
    projection = np.reshape(proj, (32,32), order='F')
    # create a figure with one row of two subplots
    figure, axs = plt.subplots(ncols=2)
    # title the subplots
    axs[0].set_title("Original")
    axs[1].set_title("Projection")
    # render images in respective subplots
    first_subplot = axs[0].imshow(original, aspect='equal')
    second_subplot = axs[1].imshow(projection, aspect='equal')
    # create a colorbar for each image
    figure.colorbar(first_subplot, ax=axs[0])
    figure.colorbar(second_subplot, ax=axs[1])
    # render plots
    figure.show()
