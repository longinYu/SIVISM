import numpy as np
import scipy.io


def density_estimation(m1, m2):
        x_min, x_max = m1.min(), m1.max()
        y_min, y_max = m2.min(), m2.max()
        X, Y = np.mgrid[x_min : x_max : 100j, y_min : y_max : 100j]                                                     
        positions = np.vstack([X.ravel(), Y.ravel()])                                                       
        values = np.vstack([m1, m2])                                                                        
        kernel = scipy.stats.gaussian_kde(values)                                                                 
        Z = np.reshape(kernel(positions).T, X.shape)
        return X, Y, Z