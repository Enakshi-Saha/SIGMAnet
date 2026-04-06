from __future__ import print_function
import time
from .timer import Timer
import sys
import os
import pandas as pd
import numpy as np
import math
from . import io
from netZooPy.dragon import *      # To load DRAGON
from scipy.stats import norm # To get normal quantiles
from scipy.stats import false_discovery_control as fdr # To get adjusted p-values
from sklearn.metrics import roc_auc_score, f1_score # To compute AUROC and F1
from scipy.stats import pearsonr
from sklearn.covariance import OAS, ShrunkCovariance, LedoitWolf, GraphicalLassoCV


class Siren():
    """
    siren


    Parameters
    ----------

            expression_file : str
                Path to file containing the gene expression data.
            methylation_file : str
                Path to file containing the methylation data.
            output_folder: str
                folder where to save the results
            delta: float
                posterior weight between 0 and 1 (If None (default) delta is tuned empirically from data)

    Notes
    ------

    Toy data:The example gene expression and methylation data that we have available here contains
    gene expression and methylation profiles for different samples in the columns.
    This is a small simulated excample.
    We provided these "toy" data so that the user can test the method.


    Sample siren results:\b
        - Node1   Node2   Weight\n
        - gene1 cpg1	0.0	-0.951416589143\n
        - gene1 cpg2	0.0	-0.904241609324\n
        .
        .
        .
        - gene2 cpg1	0.0	-0.951416589143\n
        - gene2 cpg2	0.0	-0.904241609324\n

    Authors: Enakshi Saha
    """

    def __init__(
            self,
            expression_file,
            methylation_file
    ):
        """Intialize instance of siren class and load data."""

        self.expression_file = expression_file
        self.methylation_file = methylation_file

        # data read
        self.samples = None
        self.n_samples = None
        self.expression_data = None
        self.expression_genes = None
        self.expression_samples = None
        self.methylation_data = None
        self.methylation_probes = None
        self.methylation_samples = None
        # prepare all the data
        print('siren: preparing expression and methylation')
        self._prepare_data()
        self.delta = None
        self.sirens = []
        self.precisions = []
        self.pvals = []
        self.adjPvals = []

    ########################
    ### METHODS ############
    ########################
    def _prepare_data(self):
        with Timer("Reading expression data..."):
            # Read expression
            self.expression_data, self.expression_genes = io.prepare_data(
                self.expression_file, samples=self.samples
            )

        with Timer("Reading methylation data..."):
            # Read expression
            self.methylation_data, self.methylation_probes = io.prepare_data(
                self.methylation_file, samples=self.samples
            )

            self.expression_samples = self.expression_data.columns.tolist()
            self.methylation_samples = self.methylation_data.columns.tolist()

            self.expression_data = self.expression_data.T
            self.methylation_data = self.methylation_data.T

    def run_siren(self, keep_in_memory=False, output_fmt=".hdf", output_folder='./siren_output/',
                   delta=None, precision='single', sample_names=[]):
        """siren algorithm

        Args:
            output_folder (str, optional): output folder. If an empty string is passed the matrix is automatically kept
            in memory, overwriting the value of keep_in_memory
            output_fmt (str, optional): format of output matrix. By default it is an hdf file, can be a txt or csv.
            keep_in_memory (bool, optional): if True, the partial correlation matrix is kept in memory, otherwise it is
            discarded after saving.
            only when the number of genes is not very big to avoid saving huge matrices.
            delta (float, optional): delta parameter. If default (None) delta is trained, otherwise pass a value.
            precision (str, optional): matrix precision, defaults to single precision.
        """

        siren_start = time.time()

        # first let's reorder the expression data

        if precision == 'single':
            atype = 'float32'
        elif precision == 'double':
            atype = 'float64'
        else:
            sys.exit('Precision %s unknonwn' % str(precision))

        # sort expression and methylation data
        self.expression_data = self.expression_data.astype(atype)
        self.methylation_data = self.methylation_data.astype(atype)
        self.output_fmt = output_fmt
        self.output_folder = output_folder
        # If output folder is an empty string, keep the matrix in memory and don't save it to disk
        # Otherwise the output folder can be created and the matrix saved
        if self.output_folder == '':
            keep_in_memory = True
        else:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

        # Append both omics data
        fulldata = np.append(self.expression_data, self.methylation_data, axis=1).T

        # Compute centered omics data
        z = fulldata - np.mean(fulldata, axis=0)

        # Compute covariance matrix from the rest of the data, leaving out sample
        covariance_matrix = np.cov(fulldata)

        # Compute posterior weight delta from data
        delta = 1/ (3
            + 2 * covariance_matrix.diagonal().mean()/np.sqrt(covariance_matrix.diagonal().var()))

        self.delta = delta
        self.covariance_matrix = covariance_matrix

        # Estimate OAS precision matrix
        oas = OAS(store_precision=False, assume_centered=False)
        oas.fit(fulldata.T)
        
        # Custom shrinkage parameter for individual-specific precision
        l = oas.shrinkage_
        m = (1-3*delta)/(1-delta)
        l = l * m
        
        # Shrink toward diag(S)
        # S = np.cov(fulldata)  # (p, p)
        # Sigma = (1 - l) * S + l * np.diag(np.diag(S))
        # pop_precision = np.linalg.inv(Sigma)

        # Shrink using ShrunkCovariance with custom lambda
        shrunk_cov = ShrunkCovariance(shrinkage=l, store_precision=True, assume_centered=False)
        shrunk_cov.fit(fulldata.T)
        pop_precision = shrunk_cov.precision_

        print('siren: We are starting to compute the networks...')
        if sample_names == []:
            sample_names = self.expression_samples
            sample_names = set(sample_names).intersection(set(self.methylation_samples))
        else:
            different = set(sample_names).difference(set(self.expression_samples).union(set(self.methylation_samples)))
            sample_names = set(sample_names).intersection(set(self.expression_samples))
            sample_names = set(sample_names).intersection(set(self.methylation_samples))
            if len(different) > 0:
                print('WARNING: some of the sample names are not in the expression data')
                print('\tMissing:')
                print('\t' + str(different))
                print('\tUsing:')
                print('\t' + str(sample_names))

        for s, sample in enumerate(sample_names):
            sample_start = time.time()
            # first run siren
            print('siren: network for sample %s' % str(sample))
            if keep_in_memory:
                result_siren, result_precision, result_pval_precision, result_pval_precision_adjusted = self.compute_individual_siren(
                    fulldata, pop_precision, z, s, sample, delta, keep_in_memory)
                self.sirens.append(result_siren)
                self.precisions.append(result_precision)
                self.pvals.append(result_pval_precision)
                self.adjPvals.append(result_pval_precision_adjusted)
            else:
                self.compute_individual_siren(fulldata, pop_precision, z, s, sample, delta, keep_in_memory)

        if keep_in_memory:
            return self

    def compute_individual_siren(self, fulldata, pop_precision, z, s, sample, delta, keep_in_memory):
        """Runs siren on one sample. All samples are saved separately.

        Args:
            fulldata: combined expression and methylation
            pop_precision: population level precision matrix
            z: centered omics data for the sample
            s, sample: sample index and name
            delta: delta parameter
            output_folder (str, optional): _description_.
        """

        # mask_include = [True] * fulldata.shape[1]
        # mask_include[s] = False

        print('siren: computing network for sample %s' % str(sample))
        # Compute covariance matrix from the rest of the data, leaving out sample
        # covariance_matrix = np.cov(fulldata[:, mask_include])

        # Compute posterior weight delta from data
        # if delta == None:
        #    delta = 1 / (3 + 2 * np.sqrt(covariance_matrix.diagonal()).mean()
        #                 / covariance_matrix.diagonal().var())
        # else:
        #    assert type(delta) == float

        delta = self.delta

        # Compute sample-specific precision matrix
        M = np.outer(z[:, s], z[:, s])
        df = 1 / delta + self.covariance_matrix.shape[0] + 1
        numerator = pop_precision @ M @ pop_precision * delta / (1 - delta) ** 2
        denominator = 1 + delta / (1 - delta) * np.inner(z[:, s], pop_precision @ z[:, s])
        ssprecision = pop_precision / (1 - delta) - numerator / denominator

        # Compute sample-specific partial correlation
        p = ssprecision.shape[0]
        A = np.sqrt(np.zeros((p, p)) + np.diag(ssprecision))
        ssdragon = -ssprecision / A / A.T
        ssdragon = ssdragon - np.diag(np.diag(ssdragon))

        labels = list(self.expression_data.columns) + list(self.methylation_data.columns)
        ssdragon = pd.DataFrame(ssdragon, index = labels, columns = labels)

        # Compute p-value
        pval_precision = self.compute_precision_pvalue_normal(ssprecision, df)
        # Compute FDR-corrected p-value using Benjamini-Hochberg
        pval_precision_adjusted = self.benjamini_hochberg(pval_precision)

        if not keep_in_memory:
            print('Saving siren for sample %s' % (str(sample)))
            ssdragon = pd.DataFrame(ssdragon)
            ssprecision = pd.DataFrame(ssprecision)
            pval_precision = pd.DataFrame(pval_precision)
            pval_precision_adjusted = pd.DataFrame(pval_precision_adjusted)
            sfolder = self.output_folder + './siren/'
            pfolder = self.output_folder + './precision/'
            pvalfolder = self.output_folder + './pval/'
            adjPvalfolder = self.output_folder + './adjPval/'
            if not os.path.exists(sfolder):
                os.makedirs(sfolder)
            if not os.path.exists(pfolder):
                os.makedirs(pfolder)
            if not os.path.exists(pvalfolder):
                os.makedirs(pvalfolder)
            if not os.path.exists(adjPvalfolder):
                os.makedirs(adjPvalfolder)

            output_fn_siren = sfolder + 'siren_' + str(sample) + self.output_fmt
            output_fn_precision = pfolder + 'precision_' + str(sample) + self.output_fmt
            output_fn_pval = pvalfolder + 'pval_' + str(sample) + self.output_fmt
            output_fn_adjPval = adjPvalfolder + 'adjPval_' + str(sample) + self.output_fmt
            if self.output_fmt == '.h5':
                ssdragon.to_hdf(output_fn_siren, key='siren', index=False)
                ssprecision.to_hdf(output_fn_precision, key='precision', index=False)
                pval_precision.to_hdf(output_fn_pval, key='pval', index=False)
                pval_precision_adjusted.to_hdf(output_fn_adjPval, key='adjPval', index=False)
            elif self.output_fmt == '.csv':
                ssdragon.to_csv(output_fn_siren, index=False)
                ssprecision.to_csv(output_fn_precision, index=False)
                pval_precision.to_csv(output_fn_pval, index=False)
                pval_precision_adjusted.to_csv(output_fn_adjPval, index=False)
            elif self.output_fmt == '.txt':
                ssdragon.to_csv(output_fn_siren, index=False, sep='\t')
                ssprecision.to_csv(output_fn_precision, index=False, sep='\t')
                pval_precision.to_csv(output_fn_pval, index=False, sep='\t')
                pval_precision_adjusted.to_csv(output_fn_adjPval, index=False, sep='\t')

            else:
                print('WARNING: output format (%s) not recognised. We are saving in hdf' % str(self.output_fmt))
                ssdragon.to_hdf(output_fn_siren, key='siren', index=False)
                ssprecision.to_hdf(output_fn_precision, key='precision', index=False)
                pval_precision.to_hdf(output_fn_pval, key='pval', index=False)
                pval_precision_adjusted.to_hdf(output_fn_adjPval, key='adjPval', index=False)
        else:
            return ssdragon, ssprecision, pval_precision, pval_precision_adjusted

    def compute_precision_pvalue_normal(self, ssprecision, df):
        """
            Compute element-wise p-value

            Parameters:
            ssprecision (2D array-like): sample-specific precision matrix
            df: degrees of freedom of Wishart distribution

            Returns:
            np.ndarray: p-values matrix, same shape as input.
        """

        # Extract the diagonal elements
        diag_elements = np.diag(ssprecision)

        # Create an outer product of the diagonal elements
        outer_diag_product = np.outer(diag_elements, diag_elements)

        v = np.sqrt(outer_diag_product) / df # under null, off-diagonals of ssprecision = 0

        v = np.divide(ssprecision , v)

        # Two-sided p-value: 2 * (1 - CDF(|z|)) = 2 * SF(|z|)
        p_val = 2 * (1 - norm.cdf(np.abs(v)))

        return p_val

    def benjamini_hochberg(self, pvals):
        """
        Vectorized Benjamini-Hochberg correction for a 2D matrix of p-values.

        Parameters:
        pvals (2D array-like): Input matrix of p-values.

        Returns:
        np.ndarray: Adjusted p-values matrix, same shape as input.
        """
        shape = pvals.shape  # save original shape
        flat_pvals = pvals.flatten()  # flatten to 1D

        # Apply BH correction
        adj_pvals = fdr(flat_pvals, method='bh')

        # Reshape back to 2D
        return adj_pvals.reshape(shape)

def simulate_siren_data_removeOnly(eta11, eta12, eta22, p1, p2, epsilon, n, mix_prop, seed):
    np.random.seed(seed)
    Theta = np.identity(p1 + p2)
    n11 = int(np.around(p1 * (p1 - 1) / 2 * eta11))
    n12 = int(np.around(p1 * p2 * eta12))
    n22 = int(np.around(p2 * (p2 - 1) / 2 * eta22))
    # print("n11="+str(n11)+", n12="+str(n12)+", n22="+str(n22))
    IDs = np.cumsum([0, p1, p2])
    n11_IDs = np.random.choice(range(int(p1 * (p1 - 1) / 2)), size=n11, replace=False)
    n12_IDs = np.random.choice(range(int(p1 * p2)), size=n12, replace=False)
    n22_IDs = np.random.choice(range(int(p2 * (p2 - 1) / 2)), size=n22, replace=False)
    Theta11 = Theta[IDs[0]:IDs[1], IDs[0]:IDs[1]]
    Theta12 = Theta[IDs[0]:IDs[1], IDs[1]:IDs[2]]
    Theta22 = Theta[IDs[1]:IDs[2], IDs[1]:IDs[2]]
    Theta11_vec = Theta11[np.triu_indices(p1, 1)]
    Theta12_vec = Theta12.flatten()
    Theta22_vec = Theta22[np.triu_indices(p2, 1)]

    Theta11_vec[n11_IDs] = np.random.uniform(-1., 1., size=len(n11_IDs))
    Theta12_vec[n12_IDs] = np.random.uniform(-1., 1., size=len(n12_IDs))
    Theta22_vec[n22_IDs] = np.random.uniform(-1., 1., size=len(n22_IDs))

    Theta11[np.triu_indices(p1, 1)] = Theta11_vec
    Theta22[np.triu_indices(p2, 1)] = Theta22_vec
    Theta[IDs[0]:IDs[1], IDs[1]:IDs[2]] = Theta12_vec.reshape((p1, p2))
    Theta[IDs[0]:IDs[1], IDs[0]:IDs[1]] = Theta11
    Theta[IDs[1]:IDs[2], IDs[1]:IDs[2]] = Theta22
    Theta = Theta + Theta.T - np.identity(p1 + p2)
    Theta = Theta - np.identity(p1 + p2) + np.diag(np.sum(abs(Theta), axis=0) + 0.0001)
    A = np.zeros((p1 + p2, p1 + p2)) + np.sqrt(np.diag(Theta))
    Theta1 = Theta / A / A.T
    ### Generate Theta2 by removing 10% of nonzero off-diagonal entries
    Theta2 = Theta1.copy()
    off_diag_indices = np.triu_indices_from(Theta2, k=1)
    nonzero_indices = np.where(Theta2[off_diag_indices] != 0)[0]
    n_remove = max(1, int(0.1 * len(nonzero_indices)))
    remove_indices = np.random.choice(nonzero_indices, size=n_remove, replace=False)

    # Set those entries and their symmetric counterparts to zero
    i_remove = off_diag_indices[0][remove_indices]
    j_remove = off_diag_indices[1][remove_indices]
    Theta2[i_remove, j_remove] = 0.
    Theta2[j_remove, i_remove] = 0.

    # Ensure Theta2 is still positive definite
    Theta2 += np.diag(np.maximum(0.0001, np.sum(np.abs(Theta2), axis=0)))

    # Get Sigma1 and Sigma2
    Sigma1 = np.linalg.inv(Theta1)
    Sigma2 = np.linalg.inv(Theta2)

    mu = np.zeros(p1 + p2)
    # Sample data
    n1 = int(round(n * (1 - mix_prop)))
    n2 = int(round(n * mix_prop))

    X1 = np.random.multivariate_normal(mean=mu, cov=Sigma1, size=n1)
    X2 = np.random.multivariate_normal(mean=mu, cov=Sigma2, size=n2)

    noise11 = np.random.normal(0, epsilon[0], (n1, p1))
    noise12 = np.random.normal(0, epsilon[1], (n1, p2))

    noise21 = np.random.normal(0, epsilon[0], (n2, p1))
    noise22 = np.random.normal(0, epsilon[1], (n2, p2))

    X11 = X1[:, IDs[0]:IDs[1]] + noise11
    X12 = X1[:, IDs[1]:IDs[2]] + noise12

    X21 = X2[:, IDs[0]:IDs[1]] + noise21
    X22 = X2[:, IDs[1]:IDs[2]] + noise22

    print(X1.shape, X2.shape)

    return (X11, X12, X21, X22, Theta1, Theta2, Sigma1, Sigma2)


def simulate_homogeneous_data(eta11, eta12, eta22, p1, p2, epsilon, n, seed):
    np.random.seed(seed)
    Theta = np.identity(p1+p2)
    n11 = int(np.around(p1*(p1-1)/2*eta11))
    n12 = int(np.around(p1*p2*eta12))
    n22 = int(np.around(p2*(p2-1)/2*eta22))
    print("n11="+str(n11)+", n12="+str(n12)+", n22="+str(n22))
    IDs = np.cumsum([0,p1,p2])
    n11_IDs = np.random.choice(range(int(p1*(p1-1)/2)), size=n11, replace=False)
    n12_IDs = np.random.choice(range(int(p1*p2)), size=n12, replace=False)
    n22_IDs = np.random.choice(range(int(p2*(p2-1)/2)), size=n22, replace=False)
    Theta11 = Theta[IDs[0]:IDs[1],IDs[0]:IDs[1]]
    Theta12 = Theta[IDs[0]:IDs[1],IDs[1]:IDs[2]]
    Theta22 = Theta[IDs[1]:IDs[2],IDs[1]:IDs[2]]
    Theta11_vec = Theta11[np.triu_indices(p1,1)]
    Theta12_vec = Theta12.flatten()
    Theta22_vec = Theta22[np.triu_indices(p2,1)]
    Theta11_vec[n11_IDs] = np.random.uniform(-1.,1.,size=len(n11_IDs))
    Theta12_vec[n12_IDs] = np.random.uniform(-1.,1.,size=len(n12_IDs))
    Theta22_vec[n22_IDs] = np.random.uniform(-1.,1.,size=len(n22_IDs))
    Theta11[np.triu_indices(p1,1)] = Theta11_vec
    Theta22[np.triu_indices(p2,1)] = Theta22_vec
    Theta[IDs[0]:IDs[1],IDs[1]:IDs[2]] = Theta12_vec.reshape((p1,p2))
    Theta[IDs[0]:IDs[1],IDs[0]:IDs[1]] = Theta11
    Theta[IDs[1]:IDs[2],IDs[1]:IDs[2]] = Theta22
    Theta = Theta + Theta.T - np.identity(p1+p2)
    Theta = Theta - np.identity(p1+p2) + np.diag(np.sum(abs(Theta), axis=0)+ 0.0001)
    A = np.zeros((p1+p2,p1+p2)) + np.sqrt(np.diag(Theta))
    Theta = Theta/A/A.T
    Sigma = np.linalg.inv(Theta)
    mu = np.zeros(p1+p2)
    X = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n)
    noise1 = np.random.normal(0, epsilon[0], (n,p1))
    noise2 = np.random.normal(0, epsilon[1], (n,p2))
    X1 = X[:,IDs[0]:IDs[1]] + noise1
    X2 = X[:,IDs[1]:IDs[2]] + noise2
    return(X1, X2, Theta, Sigma)


def simulate_heterogeneous_data_2pop(eta11, eta12, eta22, p1, p2, epsilon, n, mix_prop, seed):
    np.random.seed(seed)
    Theta = np.identity(p1 + p2)
    n11 = int(np.around(p1 * (p1 - 1) / 2 * eta11))
    n12 = int(np.around(p1 * p2 * eta12))
    n22 = int(np.around(p2 * (p2 - 1) / 2 * eta22))
    # print("n11="+str(n11)+", n12="+str(n12)+", n22="+str(n22))
    IDs = np.cumsum([0, p1, p2])
    n11_IDs = np.random.choice(range(int(p1 * (p1 - 1) / 2)), size=n11, replace=False)
    n12_IDs = np.random.choice(range(int(p1 * p2)), size=n12, replace=False)
    n22_IDs = np.random.choice(range(int(p2 * (p2 - 1) / 2)), size=n22, replace=False)
    Theta11 = Theta[IDs[0]:IDs[1], IDs[0]:IDs[1]]
    Theta12 = Theta[IDs[0]:IDs[1], IDs[1]:IDs[2]]
    Theta22 = Theta[IDs[1]:IDs[2], IDs[1]:IDs[2]]
    Theta11_vec = Theta11[np.triu_indices(p1, 1)]
    Theta12_vec = Theta12.flatten()
    Theta22_vec = Theta22[np.triu_indices(p2, 1)]

    Theta11_vec[n11_IDs] = np.random.uniform(-1., 1., size=len(n11_IDs))
    Theta12_vec[n12_IDs] = np.random.uniform(-1., 1., size=len(n12_IDs))
    Theta22_vec[n22_IDs] = np.random.uniform(-1., 1., size=len(n22_IDs))

    Theta11[np.triu_indices(p1, 1)] = Theta11_vec
    Theta22[np.triu_indices(p2, 1)] = Theta22_vec
    Theta[IDs[0]:IDs[1], IDs[1]:IDs[2]] = Theta12_vec.reshape((p1, p2))
    Theta[IDs[0]:IDs[1], IDs[0]:IDs[1]] = Theta11
    Theta[IDs[1]:IDs[2], IDs[1]:IDs[2]] = Theta22
    Theta = Theta + Theta.T - np.identity(p1 + p2)
    Theta = Theta - np.identity(p1 + p2) + np.diag(np.sum(abs(Theta), axis=0) + 0.0001)
    A = np.zeros((p1 + p2, p1 + p2)) + np.sqrt(np.diag(Theta))
    Theta1 = Theta / A / A.T
    ### Generate Theta2 by removing 10% of nonzero off-diagonal entries and doubling another 10%
    # Generate Theta2 by modifying 10% and removing 10% of nonzero off-diagonal entries
    # Copy the original matrix
    Theta2 = Theta1.copy()

    # Get off-diagonal upper triangle indices
    off_diag_indices = np.triu_indices_from(Theta2, k=1)

    # Identify zero and nonzero positions in upper triangle
    zero_indices = np.where(Theta2[off_diag_indices] == 0)[0]
    nonzero_indices = np.where(Theta2[off_diag_indices] != 0)[0]

    # Determine number of entries to modify
    n_flip_zero = max(1, int(0.1 * len(zero_indices)))
    n_flip_nonzero = max(1, int(0.1 * len(nonzero_indices)))

    # Sample indices to modify
    add_indices = np.random.choice(zero_indices, size=n_flip_zero, replace=False)
    remove_indices = np.random.choice(nonzero_indices, size=n_flip_nonzero, replace=False)

    # Make zero → nonzero (assign random value in [-1, 1])
    i_add = off_diag_indices[0][add_indices]
    j_add = off_diag_indices[1][add_indices]
    new_values = np.random.uniform(-1, 1, size=n_flip_zero)
    Theta2[i_add, j_add] = new_values
    Theta2[j_add, i_add] = new_values  # maintain symmetry

    # Make nonzero → zero
    i_remove = off_diag_indices[0][remove_indices]
    j_remove = off_diag_indices[1][remove_indices]
    Theta2[i_remove, j_remove] = 0.
    Theta2[j_remove, i_remove] = 0.


    # Ensure Theta2 is still positive definite
    Theta2 += np.diag(np.maximum(0.0001, np.sum(np.abs(Theta2), axis=0)))

    # Get Sigma1 and Sigma2
    Sigma1 = np.linalg.inv(Theta1)
    Sigma2 = np.linalg.inv(Theta2)

    mu = np.zeros(p1 + p2)
    # Sample data
    n1 = int(round(n * (1 - mix_prop)))
    n2 = int(round(n * mix_prop))

    X1 = np.random.multivariate_normal(mean=mu, cov=Sigma1, size=n1)
    X2 = np.random.multivariate_normal(mean=mu, cov=Sigma2, size=n2)

    noise11 = np.random.normal(0, epsilon[0], (n1, p1))
    noise12 = np.random.normal(0, epsilon[1], (n1, p2))

    noise21 = np.random.normal(0, epsilon[0], (n2, p1))
    noise22 = np.random.normal(0, epsilon[1], (n2, p2))

    X11 = X1[:, IDs[0]:IDs[1]] + noise11
    X12 = X1[:, IDs[1]:IDs[2]] + noise12

    X21 = X2[:, IDs[0]:IDs[1]] + noise21
    X22 = X2[:, IDs[1]:IDs[2]] + noise22

    print(X1.shape, X2.shape)

    return (X11, X12, X21, X22, Theta1, Theta2, Sigma1, Sigma2)


def simulate_heterogeneous_data_3pop(eta11, eta12, eta22, p1, p2, epsilon, n, mix_props, seed):
    """
    mix_props: tuple of 3 proportions (p1, p2, p3) that sum to 1
               e.g. (0.3, 0.3, 0.4)
    Returns: X11, X12, X21, X22, X31, X32, Theta1, Theta2, Theta3, Sigma1, Sigma2, Sigma3
    """
    assert len(mix_props) == 3, "mix_props must have 3 elements"
    assert abs(sum(mix_props) - 1.0) < 1e-6, "mix_props must sum to 1"

    np.random.seed(seed)

    def make_theta(eta11, eta12, eta22, p1, p2):
        """Generate a random precision matrix."""
        Theta = np.identity(p1 + p2)
        IDs = np.cumsum([0, p1, p2])
        n11 = int(np.around(p1 * (p1 - 1) / 2 * eta11))
        n12 = int(np.around(p1 * p2 * eta12))
        n22 = int(np.around(p2 * (p2 - 1) / 2 * eta22))
        n11_IDs = np.random.choice(range(int(p1*(p1-1)/2)), size=n11, replace=False)
        n12_IDs = np.random.choice(range(int(p1*p2)), size=n12, replace=False)
        n22_IDs = np.random.choice(range(int(p2*(p2-1)/2)), size=n22, replace=False)
        Theta11 = Theta[IDs[0]:IDs[1], IDs[0]:IDs[1]]
        Theta12 = Theta[IDs[0]:IDs[1], IDs[1]:IDs[2]]
        Theta22 = Theta[IDs[1]:IDs[2], IDs[1]:IDs[2]]
        Theta11_vec = Theta11[np.triu_indices(p1, 1)]
        Theta12_vec = Theta12.flatten()
        Theta22_vec = Theta22[np.triu_indices(p2, 1)]
        Theta11_vec[n11_IDs] = np.random.uniform(-1., 1., size=len(n11_IDs))
        Theta12_vec[n12_IDs] = np.random.uniform(-1., 1., size=len(n12_IDs))
        Theta22_vec[n22_IDs] = np.random.uniform(-1., 1., size=len(n22_IDs))
        Theta11[np.triu_indices(p1, 1)] = Theta11_vec
        Theta22[np.triu_indices(p2, 1)] = Theta22_vec
        Theta[IDs[0]:IDs[1], IDs[1]:IDs[2]] = Theta12_vec.reshape((p1, p2))
        Theta[IDs[0]:IDs[1], IDs[0]:IDs[1]] = Theta11
        Theta[IDs[1]:IDs[2], IDs[1]:IDs[2]] = Theta22
        Theta = Theta + Theta.T - np.identity(p1 + p2)
        Theta = Theta - np.identity(p1 + p2) + np.diag(np.sum(abs(Theta), axis=0) + 0.0001)
        A = np.zeros((p1+p2, p1+p2)) + np.sqrt(np.diag(Theta))
        Theta = Theta / A / A.T
        return Theta, IDs

    # Generate 3 distinct precision matrices
    # Theta1 is the base
    Theta1, IDs = make_theta(eta11, eta12, eta22, p1, p2)

    ### Generate Theta2 and Theta3 as a chain of perturbations from Theta1
    # Theta2: remove 10% of nonzero off-diagonal entries and double another 10% from Theta1
    # Theta3: remove 10% of nonzero off-diagonal entries and double another 10% from Theta2
    # This creates a gradient of increasing dissimilarity: Theta1 -> Theta2 -> Theta3
    def perturb_theta(Theta_base, seed_offset):
        rng = np.random.RandomState(seed + seed_offset)
        Theta_new = Theta_base.copy()
        off_diag_indices = np.triu_indices_from(Theta_new, k=1)
        nonzero_indices = np.where(Theta_new[off_diag_indices] != 0)[0]
        
        n_remove = max(1, int(0.1 * len(nonzero_indices)))
        n_double = max(1, int(0.1 * len(nonzero_indices)))
        
        # Sample distinct indices for removing and doubling
        chosen = rng.choice(nonzero_indices, size=n_remove + n_double, replace=False)
        remove_indices = chosen[:n_remove]
        double_indices = chosen[n_remove:]
        
        # Remove edges
        i_remove = off_diag_indices[0][remove_indices]
        j_remove = off_diag_indices[1][remove_indices]
        Theta_new[i_remove, j_remove] = 0.
        Theta_new[j_remove, i_remove] = 0.
        
        # Double edges
        i_double = off_diag_indices[0][double_indices]
        j_double = off_diag_indices[1][double_indices]
        Theta_new[i_double, j_double] *= 2
        Theta_new[j_double, i_double] *= 2
        
        # Ensure positive definite
        Theta_new += np.diag(np.maximum(0.0001, np.sum(np.abs(Theta_new), axis=0)))
        return Theta_new

    Theta2 = perturb_theta(Theta1, seed_offset=1)
    Theta3 = perturb_theta(Theta2, seed_offset=2)  # chained from Theta2

    Sigma1 = np.linalg.inv(Theta1)
    Sigma2 = np.linalg.inv(Theta2)
    Sigma3 = np.linalg.inv(Theta3)

    mu = np.zeros(p1 + p2)
    n1 = int(round(n * mix_props[0]))
    n2 = int(round(n * mix_props[1]))
    n3 = n - n1 - n2  # ensure total sums to n

    X1 = np.random.multivariate_normal(mean=mu, cov=Sigma1, size=n1)
    X2 = np.random.multivariate_normal(mean=mu, cov=Sigma2, size=n2)
    X3 = np.random.multivariate_normal(mean=mu, cov=Sigma3, size=n3)

    noise11 = np.random.normal(0, epsilon[0], (n1, p1))
    noise12 = np.random.normal(0, epsilon[1], (n1, p2))
    noise21 = np.random.normal(0, epsilon[0], (n2, p1))
    noise22 = np.random.normal(0, epsilon[1], (n2, p2))
    noise31 = np.random.normal(0, epsilon[0], (n3, p1))
    noise32 = np.random.normal(0, epsilon[1], (n3, p2))

    X11 = X1[:, IDs[0]:IDs[1]] + noise11
    X12 = X1[:, IDs[1]:IDs[2]] + noise12
    X21 = X2[:, IDs[0]:IDs[1]] + noise21
    X22 = X2[:, IDs[1]:IDs[2]] + noise22
    X31 = X3[:, IDs[0]:IDs[1]] + noise31
    X32 = X3[:, IDs[1]:IDs[2]] + noise32

    print(f"n1={n1}, n2={n2}, n3={n3}")
    return (X11, X12, X21, X22, X31, X32, Theta1, Theta2, Theta3, Sigma1, Sigma2, Sigma3)


def compute_sampleSpecific_AUC(Theta, adjPvals):
    Theta0 = ((np.abs(Theta) > 0)).astype(int).flatten()

    auc_scores = []

    for index in range(len(adjPvals)):
        adj_pval_matrix = adjPvals[index]

        # Flatten the arrays to 1D for comparison
        y_score = adj_pval_matrix.flatten()
        y_true = Theta0.flatten()

        auc = roc_auc_score(y_true, 1 - y_score)  # Lower p-val means stronger signal, so use 1 - p
        auc_scores.append(auc)

    return auc_scores

def compute_sampleSpecific_F1(Theta, adjPvals, pval_threshold = 0.05):
    Theta0 = ((np.abs(Theta) > 0)).astype(int).flatten()

    f1_scores = []

    for index in range(len(adjPvals)):
        adj_pval_matrix = adjPvals[index]
        adj_pval_matrix = ((adj_pval_matrix < pval_threshold)).astype(int).flatten()

        # Flatten the arrays to 1D for comparison
        y_score = adj_pval_matrix.flatten()
        y_true = Theta0.flatten()

        f1 = f1_score(y_true, y_score)  # Lower p-val means stronger signal, so use 1 - p
        f1_scores.append(f1)

    return f1_scores

def compute_sampleSpecific_correlation(Theta, sirens):
    Theta0 = Theta.flatten()
    corrs = []

    for siren_matrix in sirens:
        siren_matrix = siren_matrix.flatten()

        # Compute Pearson correlation
        cor, _ = pearsonr(Theta0, siren_matrix)
        corrs.append(cor)

    return corrs

def compute_sampleSpecific_frobenius(Theta, sirens):
    frobs = []

    for siren_matrix in sirens:
        # Ensure same shape
        if Theta.shape != siren_matrix.shape:
            raise ValueError(f"Shape mismatch: Theta {Theta.shape}, siren {siren_matrix.shape}")

        # Compute Frobenius norm of difference
        frob = np.linalg.norm(Theta - siren_matrix, 'fro')
        frobs.append(frob)

    return frobs

def evaluate_dragon_mixture(X1, X2, mix_prop, Theta1, Theta2):
    n = X1.shape[0]
    p1 = X1.shape[1]
    p2 = X2.shape[1]
    n1 = int(n * (1-mix_prop))
    n2 = int(n * mix_prop)
    expr1 = X1[:n1]
    expr2 = X1[n1:]
    meth1 = X2[:n1]
    meth2 = X2[n1:]
    
    lambdas1, _ = estimate_penalty_parameters_dragon(expr1, meth1)
    r1 = get_partial_correlation_dragon(expr1, meth1, lambdas1)
    adj_p_vals, p_vals = estimate_p_values_dragon(r1, n1, p1, p2, lambdas1)
    # Flatten the arrays to 1D for comparison
    y_score = adj_p_vals.flatten()
    y_score_f = ((y_score < 0.05)).astype(int).flatten()
    Theta01 = ((np.abs(Theta1) > 0)).astype(int).flatten()
    y_true1 = Theta01.flatten()
    auc1 = roc_auc_score(y_true1, 1 - y_score)
    f1_1 = f1_score(y_true1, y_score_f)
    dcor1, _ = pearsonr(Theta1.flatten(), r1.flatten())
    dfrob1 = np.linalg.norm(Theta1 - r1, 'fro')
    
    lambdas2, _ = estimate_penalty_parameters_dragon(expr2, meth2)
    r2 = get_partial_correlation_dragon(expr2, meth2, lambdas2)
    adj_p_vals, p_vals = estimate_p_values_dragon(r2, n2, p1, p2, lambdas2)
    Theta02 = ((np.abs(Theta2) > 0)).astype(int).flatten()
    y_true2 = Theta02.flatten()
    auc2 = roc_auc_score(y_true2, 1 - y_score) 
    f1_2 = f1_score(y_true2, y_score_f)
    dcor2, _ = pearsonr(Theta2.flatten(), r2.flatten())
    dfrob2 = np.linalg.norm(Theta2 - r2, 'fro')
    
    auc = auc1 * (1-mix_prop) + auc2 * mix_prop
    f1 = f1_1 * (1-mix_prop) + f1_2 * mix_prop
    dcor = dcor1 * (1-mix_prop) + dcor2 * mix_prop
    dfrob = dfrob1 * (1-mix_prop) + dfrob2 * mix_prop
    
    return auc, f1, dcor, dfrob


def evaluate_OAS_mixture(X1, X2, mix_props, Thetas):
    """
    mix_props: list of proportions e.g. [0.6, 0.4] or [0.3, 0.3, 0.4]
    Thetas: list of true precision matrices [Theta1, Theta2, ...] 
    """
    n = X1.shape[0]
    assert len(mix_props) == len(Thetas), "mix_props and Thetas must have same length"
    
    ns = [int(round(n * p)) for p in mix_props]
    ns[-1] = n - sum(ns[:-1])  # ensure total sums to n
    
    aucs, f1s, dcors, dfrobs = [], [], [], []
    start = 0
    for i, (n_sub, Theta) in enumerate(zip(ns, Thetas)):
        end = start + n_sub
        expr_sub = X1[start:end]
        meth_sub = X2[start:end]
        combined = np.append(expr_sub, meth_sub, axis=1)
        oas = OAS(store_precision=True, assume_centered=False)
        oas.fit(combined)
        r = oas.precision_
        y_true = ((np.abs(Theta) > 0)).astype(int).flatten()
        y_score = np.abs(r).flatten()
        aucs.append(roc_auc_score(y_true, y_score))
        f1s.append(f1_score(y_true, (y_score > 0).astype(int)))
        dcors.append(pearsonr(Theta.flatten(), r.flatten())[0])
        dfrobs.append(np.linalg.norm(Theta - r, 'fro'))
        start = end

    auc = sum(a * p for a, p in zip(aucs, mix_props))
    f1 = sum(f * p for f, p in zip(f1s, mix_props))
    dcor = sum(d * p for d, p in zip(dcors, mix_props))
    dfrob = sum(d * p for d, p in zip(dfrobs, mix_props))
    return auc, f1, dcor, dfrob


def evaluate_LW_mixture(X1, X2, mix_props, Thetas):
    n = X1.shape[0]
    assert len(mix_props) == len(Thetas)
    
    ns = [int(round(n * p)) for p in mix_props]
    ns[-1] = n - sum(ns[:-1])

    aucs, f1s, dcors, dfrobs = [], [], [], []
    start = 0
    for n_sub, Theta in zip(ns, Thetas):
        end = start + n_sub
        expr_sub = X1[start:end]
        meth_sub = X2[start:end]
        combined = np.append(expr_sub, meth_sub, axis=1)
        lw = LedoitWolf(store_precision=True, assume_centered=False)
        lw.fit(combined)
        r = lw.precision_
        y_true = ((np.abs(Theta) > 0)).astype(int).flatten()
        y_score = np.abs(r).flatten()
        aucs.append(roc_auc_score(y_true, y_score))
        f1s.append(f1_score(y_true, (y_score > 0).astype(int)))
        dcors.append(pearsonr(Theta.flatten(), r.flatten())[0])
        dfrobs.append(np.linalg.norm(Theta - r, 'fro'))
        start = end

    auc = sum(a * p for a, p in zip(aucs, mix_props))
    f1 = sum(f * p for f, p in zip(f1s, mix_props))
    dcor = sum(d * p for d, p in zip(dcors, mix_props))
    dfrob = sum(d * p for d, p in zip(dfrobs, mix_props))
    return auc, f1, dcor, dfrob


def evaluate_GL_mixture(X1, X2, mix_props, Thetas):
    n = X1.shape[0]
    assert len(mix_props) == len(Thetas)
    
    ns = [int(round(n * p)) for p in mix_props]
    ns[-1] = n - sum(ns[:-1])

    aucs, f1s, dcors, dfrobs = [], [], [], []
    start = 0
    for n_sub, Theta in zip(ns, Thetas):
        end = start + n_sub
        expr_sub = X1[start:end]
        meth_sub = X2[start:end]
        combined = np.append(expr_sub, meth_sub, axis=1)
        gl = GraphicalLassoCV(assume_centered=False)
        gl.fit(combined)
        r = gl.precision_
        y_true = ((np.abs(Theta) > 0)).astype(int).flatten()
        y_score = np.abs(r).flatten()
        aucs.append(roc_auc_score(y_true, y_score))
        f1s.append(f1_score(y_true, (y_score > 0).astype(int)))
        dcors.append(pearsonr(Theta.flatten(), r.flatten())[0])
        dfrobs.append(np.linalg.norm(Theta - r, 'fro'))
        start = end

    auc = sum(a * p for a, p in zip(aucs, mix_props))
    f1 = sum(f * p for f, p in zip(f1s, mix_props))
    dcor = sum(d * p for d, p in zip(dcors, mix_props))
    dfrob = sum(d * p for d, p in zip(dfrobs, mix_props))
    return auc, f1, dcor, dfrob
