import pandas as pd
import numpy as np
import itertools
from numba import njit, jit

"""
------------------------------------------
General SHAP computation helper functions.
------------------------------------------
"""


@njit
def find_index(array, item):
    """
    Accelerated index finder.
    """
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx[0]


def computeb1b2(x, w, o, i, pre_idx, order=-1, prng=None):
    """
    Compute b1 and b2 for order sample o, instance sample w, and feature index i.

    Parameters
    ----------
    w : numpy array
        array of size n with feature values of instance (w)
    o : numpy array
        array of size n with order of features
    i : int
        feature index
    pre_idx : numpy array
        arrangement of feature indices
    """
    pos_i = find_index(o, i)  # pos_i = np.where(o == i)[0][0]
    idx = pre_idx[pos_i + 1:]  # positions succeeding i
    o_idx = o[idx]  # features succeeding i
    b1 = x.copy()
    b1[o_idx] = w[o_idx]  # fill features succeeding i with w
    b2 = b1.copy()
    b2[i] = w[i]  # change x_i to w_i
    return b1, b2


"""
-----------------------------------
Exact computation helper functions.
-----------------------------------
"""


def retrieve_instances(mc, X):
    """
    Retrieve all
    """
    if mc == 'uniform-cat':
        z = [X[c].unique() for c in X.columns]
        instances = list(itertools.product(*z))
    elif mc == 'training':
        instances = X.as_matrix()
    return instances


def retrieve_permutations(n):
    permutations = list(itertools.permutations(range(n)))
    return permutations


"""
-----------------------------------
Adaptive Sampling helper functions.
-----------------------------------
"""


def update(existingAggregate, newValue):
    """
    Update the count, mean, and mean square.

    Welford's online algorithm for calculating variance.

    existingAggretate : tuple
        (count, mean, M2)
    newValue : float
        f(b1) - f(b2) for newest sample
    """
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    sampleVariance = M2 / (count - 1)
    return count, mean, M2, sampleVariance


"""
-------------
SHAP classes.
-------------
"""


class Generator:
    """
    Generator for instaces (w)
    """

    def __init__(self, X, mc):
        """
        Paramaters
        ----------
        X : pandas DataFrame
            Training data for sampling instances (w)
        mc : string
            - 'training' : sample instances (w) from training data
        """
        self.mc = mc
        self.feature_generators = {}
        self.columns = X.columns.tolist()
        if mc == 'training':
            self.X = X
        else:
            raise ValueError("'%s' is an invalid Monte Carlo sampling strategy." % self.mc)
        return

    def sample(self, n, seed=1, sample_type='dict', replace=True):
        """
        Sample n ranodm instances (w).
        Parameters
        ----------
        n : int
            number of samples
        seed : int
            pseudorandom seed
        replacement : Boolean
            sample with replacement (True) or without replacement (False)
        Returns
        -------
        samples : numpy array
            two dimensional numpy array of feature * instances
        """
        if self.mc == 'training':
            samples = self.X.sample(n=n, random_state=seed, replace=replace)
        return samples


class Values:
    """
    Store SHAP values and samples.
    """

    def __init__(self, shap_values, abs_shap_values, samples=None):
        self.shap_values = shap_values
        self.abs_shap_values = abs_shap_values
        self.samples = samples
        return


class Explainer:
    """
    Object that can be called to compute SHAP values. Stores training data, classifier and sampling parameter.
    """

    def __init__(self, X, mc='training', f=None):
        """
        Paramaters
        ----------
        X : pandas DataFrame
            Training data for sampling instances (w)
        mc : string
            - 'training' : sample instances (w) from training data
        """
        self.X = X
        self.mc = mc
        self.f = f
        return

    def standard(self, x, m, f=None, X=None, mc=None, seed=1, verbose=False, return_samples=False):

        """
        Naive Monte Carlo approximation of SHAP values.

        Parameters
        ----------
        x : numpy array
            numpy array containing all feature values, features must be ordered according to dataframe
        f : object
            model should have function '.predict_proba()' that takes as input an instance and as
            output a probability for the positive class
        m : int
            number of samples for each feature
        X : pandas DataFrame
            training dataset
        mc : string
            Monte Carlo sampling strategy that indicates how random instances are sampled.
            The sampling strategy affects the computation of the conditional expectation.
            'training' : random instances will be sampled from the training data
        seed : int
            seed used for generating random instances and choosing random orders
        verbose : Boolean
            controls verbosity
        return_samples : Boolean
            returning samples that were used to commpute SHAP values to allow for SHAP-ICE and SHAP-ICC plots.
        """

        # Retrieve explainer variables
        X, mc, f = self.X, self.mc, self.f

        # Initialize vars
        features = np.arange(len(X.columns)).astype(int)  # numpy array with feature indexes
        n = len(features)
        chi = Generator(X=X, mc=mc)
        phi = {}
        abs_phi = {}
        pre_idx = np.arange(len(features))

        # Sample all permutations (o)
        prng = np.random.RandomState(seed=seed)
        permutations = [prng.permutation(range(n)) for i in range(m * n)]

        # Sample all instances (w)
        samples = np.array(chi.sample(n=m * n, seed=seed, sample_type='array'))

        # TEMP
        temp_results = {}

        # Compute all b1 and b2
        b1_all = [0] * (m * n)  # initialize list with all b1's
        b2_all = [0] * (m * n)  # initialize list with all b2's
        for i, index_n in zip(features, range(n)):  # for each feature
            temp_feature_results = []
            for w, o, index_m in zip(samples[index_n * m:(index_n + 1) * m],
                                     permutations[index_n * m:(index_n + 1) * m], range(m)):
                # for each sample index_m, consisting of instance w and order o:
                b1, b2 = computeb1b2(x, w, o, i, pre_idx)
                all_index = index_n * m + index_m
                b1_all[all_index] = b1
                b2_all[all_index] = b2
                # TEMP
                temp_feature_results.append({'o': tuple(o), 'w': w, 'b1': b1, 'b2': b2, 'v': w[i]})
            temp_results[i] = pd.DataFrame(temp_feature_results)

        # Make predictions for instances b1 and b2
        predictions = np.array(f.predict_proba(b1_all + b2_all))[:, 1]
        if verbose:
            print("Average predictions b1/b2: %.5f" % (np.mean(predictions)))

        # Compute Shapley value based on marginal contributions
        for i, j in zip(X.columns, features):
            b1_i = predictions[(j * m):(j + 1) * m]
            b2_i = predictions[(n * m + j * m):(n * m + (j + 1) * m)]
            phi[i] = sum(b1_i - b2_i) / m
            abs_phi[i] = sum([abs(diff) for diff in (b1_i - b2_i)]) / m

            # TEMP
            temp_results[j]['f(b1)'] = b1_i
            temp_results[j]['f(b2)'] = b2_i
            temp_results[j]['c'] = b1_i - b2_i
        if return_samples:
            return Values(phi, abs_phi, samples=temp_results)
        else:
            return Values(phi, abs_phi)