
import numpy as np
from scipy.linalg import cholesky


def sample_wishart(scale, df, rng):
    dim = scale.shape[0]
    n_tril = dim * (dim - 1) // 2
    c_matrix = cholesky(scale, lower=True)
    covariances = rng.normal(size=n_tril) + 1j * rng.normal(size=n_tril)
    # diagonal elements follow random gamma distribution (according to Nagar and Gupta, 2011)
    variances = np.r_[[rng.gamma(df - dim + i, scale=1, size=1) ** 0.5 for i in range(dim)]]
    a_matrix = np.zeros(c_matrix.shape, dtype=complex)
    # input the covariances
    tril_idx = np.tril_indices(dim, k=-1)
    a_matrix[tril_idx] = covariances
    # Input the variances
    a_matrix[np.diag_indices(dim)] = variances.astype(complex, copy=False)[:, 0]
    # build matrix
    ca_matrix = np.dot(c_matrix, a_matrix)
    return np.dot(ca_matrix, ca_matrix.conjugate().T) / df
