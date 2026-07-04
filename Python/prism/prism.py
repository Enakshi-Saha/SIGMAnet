from __future__ import print_function
import math
import time
import pandas as pd
from .timer import Timer
import numpy as np
from netZooPy.panda.panda import Panda
from netZooPy.panda import calculations as calc
from . import io
import sys
import os
import glob
import pandas as pd


def get_n_matrix(df):
    """Pairwise sample-count matrix accounting for missing values.

    Parameters
    ----------
    df : pd.DataFrame
        genes x samples, may contain NaNs.

    Returns
    -------
    np.ndarray, shape (g, g)
    """
    N = len(df.columns)
    nn = N - df.isna().sum(axis=1).values[:, np.newaxis]
    nr = np.repeat(nn, len(nn), axis=1)
    return np.minimum(nr, nr.T)


def estimate_delta_jackknife(expression_data):
    """Closed-form jackknife estimate of delta.

    Parameters
    ----------
    expression_data : pandas.DataFrame or np.ndarray, shape (g, n)
        Genes (rows) x samples (columns).

    Returns
    -------
    float
        Estimated delta.
    """
    X = np.asarray(expression_data, dtype='float64')
    g, n = X.shape

    s_kk = X.var(axis=1, ddof=1)  # shape (g,)

    s1 = X.sum(axis=1, keepdims=True)
    s2 = (X ** 2).sum(axis=1, keepdims=True)

    s1_loo = s1 - X
    s2_loo = s2 - X ** 2
    mean_loo = s1_loo / (n - 1)
    var_loo = (s2_loo - (n - 1) * mean_loo ** 2) / (n - 2)

    eta = var_loo.var(axis=1, ddof=1)

    numerator = 2 * np.sum(s_kk ** 2)
    denominator = np.sum(eta)

    nu = 3 + numerator / denominator
    delta = 1.0 / nu
    return delta


def compute_sample_coexpression(sample, expression_data, expression_mean,
                                 covariance_matrix, delta, n_matrix_full):
    """Compute the sample-specific coexpression matrix for one sample.

    This is the single shared implementation used by Prism.compute(), so
    any future correction (e.g. missing-data handling) only needs to be
    made here.

    Parameters
    ----------
    sample : str
        Column name of the sample in expression_data.
    expression_data : pd.DataFrame
        genes x samples, may contain NaNs.
    expression_mean : np.ndarray, shape (g, 1)
        Grand mean per gene.
    covariance_matrix : np.ndarray, shape (g, g)
        Population covariance S.
    delta : float
        Shrinkage weight.
    n_matrix_full : np.ndarray, shape (g, g)
        Pairwise sample-count matrix for the full dataset.

    Returns
    -------
    pd.DataFrame, shape (g, g)
    """
    names = expression_data.index.tolist()
    touse = [s for s in expression_data.columns if s != sample]

    centered_sample = (expression_data - expression_mean).loc[:, sample]

    sscov = delta * np.outer(centered_sample, centered_sample) + (1 - delta) * covariance_matrix
    sscov = np.array(sscov)

    diag = np.sqrt(np.diag(np.diag(sscov)))
    diag = np.array(diag)
    zero_idx = np.where(np.diag(diag) == 0)[0]
    for i in zero_idx:
        diag[i, i] = 1
    sds = np.linalg.inv(diag)
    coexpression = sds @ sscov @ sds

    n_loo = get_n_matrix(expression_data.loc[:, touse])
    nmatrix = n_matrix_full - n_loo

    coexpression = pd.DataFrame(
        data=np.multiply(nmatrix, coexpression), index=names, columns=names
    )
    # NOTE: missing-data fallback (Section 2.5 of the paper) not yet applied
    # here. Currently unobserved gene-pairs are zeroed out via nmatrix;
    # per the derivation, they should instead fall back to the prior
    # correlation. TODO: fix in this function only -- Prism.compute()
    # and PrismGRN both inherit the fix automatically once applied here.
    return coexpression


class Prism:
    """Core PRISM estimator: expression data -> sample-specific coexpression.

    Handles delta calibration (jackknife-based, by default) and computes
    one coexpression matrix per sample. This class owns all covariance
    computation; PrismMultiomic and PrismGRN build on top of it rather
    than duplicating this logic.

    Parameters
    ----------
    expression_file : str or pandas.DataFrame
        Genes (rows) x samples (columns); path to a tab-separated file
        (no header) or a DataFrame.
    delta : float
        Shrinkage weight in (0, 1]. Ignored if tune_delta=True.
    tune_delta : bool
        Calibrate delta from data via the jackknife estimator (default)
        instead of using the supplied value.
    precision : {'single', 'double'}
        Floating point precision for the expression data.

    Authors: Enakshi Saha
    """

    def __init__(self, expression_file, delta=0.1, tune_delta=True, precision='single'):
        self.expression_data = self._load(expression_file, precision)
        self.samples = self.expression_data.columns.tolist()
        self.genes = self.expression_data.index.tolist()

        self.n_matrix_full = get_n_matrix(self.expression_data)
        self.expression_mean = np.nanmean(self.expression_data.values, axis=1, keepdims=True)
        self.covariance_matrix = self.expression_data.T.cov().values

        self.delta = estimate_delta_jackknife(self.expression_data) if tune_delta else delta

    def _load(self, expression_file, precision):
        atype = 'float32' if precision == 'single' else 'float64'
        if isinstance(expression_file, pd.DataFrame):
            return expression_file.astype(atype)
        data, _ = io.prepare_expression(expression_file)
        return data.astype(atype)

    def compute(self, keep_output=False, output_folder='./prism_coexpress/'):
        """Estimate the sample-specific coexpression matrix for every sample.

        Parameters
        ----------
        keep_output : bool
            Save each sample's matrix to output_folder if True.
        output_folder : str

        Returns
        -------
        dict[str, pandas.DataFrame]
            Sample name -> (genes x genes) coexpression matrix.
        """
        if keep_output and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        results = {}
        for sample in self.samples:
            coexp = compute_sample_coexpression(
                sample, self.expression_data, self.expression_mean,
                self.covariance_matrix, self.delta, self.n_matrix_full
            )
            results[sample] = coexp
            if keep_output:
                coexp.to_csv(f'{output_folder}coexpression_{sample}.txt', sep='\t')

        return results


class PrismMultiomic(Prism):
    """Prism with a nonparanormal (Gaussian copula) transform applied to
    the expression data before coexpression estimation.

    Intended for non-Gaussian omics modalities (e.g. methylation,
    proteomics, CNV segment means). Reuses Prism.compute() unchanged --
    only data preparation differs, via rank-based marginal Gaussianization
    (Liu et al. 2009) applied prior to delta calibration and covariance
    estimation.

    Authors: Enakshi Saha
    """

    def __init__(self, expression_file, delta=0.1, tune_delta=True, precision='single'):
        raw = self._load(expression_file, precision)
        transformed = self._nonparanormal_transform(raw)
        super().__init__(transformed, delta=delta, tune_delta=tune_delta, precision=precision)

    def _nonparanormal_transform(self, expression_data):
        """Rank-based Gaussianization and Spearman/Kendall-to-Pearson
        correlation conversion (Liu et al. 2009).

        TODO: implement. Should compute the correlation matrix via
        Spearman's rho or Kendall's tau on the raw data (invariant to
        monotone marginal transforms), convert to the implied Pearson
        correlation via r = 2*sin(pi/6 * rho) or r = sin(pi/2 * tau),
        and return data/statistics consistent with what Prism.__init__
        expects downstream.
        """
        raise NotImplementedError(
            "Nonparanormal transform not yet implemented."
        )


class PrismGRN:
    """Downstream use case: PRISM coexpression estimation + PANDA GRN
    inference, one network per sample using a sample-specific motif prior.

        1. Reading in input data (expression, motif prior table, TF PPI data)
        2. Preparing motif prior universe
        3. Estimating sample-specific coexpression with Prism
        4. Running PANDA with a different prior for each sample

    This class delegates all coexpression computation to an internal
    Prism instance; it does not implement any covariance math itself.

    Parameters
    ----------

            expression_file : str
                Path to file containing the gene expression data or pandas dataframe. By default, the expression file does not have a header, and the cells ares separated by a tab.
            priors_table_file : str
                Path to file containing a table where each samples is linked to its own motif prior file
            ppi_file : str
                Path to file containing the PPI data. or pandas dataframe.
                The PPI can be symmetrical, if not, it will be transformed into a symmetrical adjacency matrix.
            mode_process : str
                The input data processing mode.
                - 'legacy': refers to the processing mode in netZooPy<=0.5
                - (Default)'union': takes the union of all TFs and genes across priors and fills the missing genes in the priors with zeros.
                - 'intersection': intersects the input genes and TFs across priors and removes the missing TFs/genes.
            mode_priors: str
                The prior data processing
            prior_tf_col: str
                name of the tf column in the prior files
            prior_gene_col: str
                name of the gene column in the prior files
            output_folder: str
                folder where to save the results
            delta: float
                posterior weight between 0 and 1 (Default to 0.3)
            tune_delta: boolean
                if true, the posterior weight (delta) for the estimation of the single sample coexpression is estimated from data

    Notes
    ------

    Toy data:The example gene expression data that we have available here contains gene expression profiles
    for different samples in the columns. Of note, this is just a small subset of a larger gene
    expression dataset. We provided these "toy" data so that the user can test the method.

    Sample PANDA results:\b
        - TF    Gene    Motif   Force\n
        - CEBPA AACSL	0.0	-0.951416589143\n
        - CREB1 AACSL	0.0	-0.904241609324\n
        - DDIT3 AACSL	0.0	-0.956471642313\n
        - E2F1  AACSL	1.0	3.685316051\n
        - EGR1  AACSL	0.0	-0.695698519643

    References
    ----------
    .. [1]__

    Authors: Enakshi Saha
    """

    def __init__(
        self,
        expression_file,
        priors_table_file,
        ppi_table_file=None,
        ppi_file=None,
        mode_process="union",
        mode_priors="union",
        prior_tf_col=0,
        prior_gene_col=1,
        output_folder='./prism/'
    ):
        """Initialize PrismGRN and load motif/PPI/priors data."""

        self.expression_file = expression_file
        self.priors_table_file = priors_table_file
        self.ppi_table_file = ppi_table_file
        self.ppi_file = ppi_file
        if self.ppi_file:
            self.ppi_mode = 'motif'
        else:
            self.ppi_mode = 'sample'
        self.mode_process = mode_process
        self.mode_priors = mode_priors
        self.prior_tf_col = prior_tf_col
        self.prior_gene_col = prior_gene_col
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # data read
        self.samples = None
        self.n_samples = None
        self.prior_dict = None
        self.expression_data = None
        self.expression_genes = None

        # we need to keep track of all the names in the expression and motif data
        self.priors_tfs = None
        self.priors_genes = None

        # dictionaries mapping sample:prior_file and prior_file:[samples]
        self.sample2prior_dict = None
        self.prior2sample_dict = None

        # SORTED LIST OF TFS AND GENES
        self.universe_tfs = None
        self.universe_genes = None
        self.gene2idx = None
        self.tf2idx = None

        # the Prism instance used for coexpression estimation, built once
        # self.prepare_data / run() has restricted expression_data to
        # universe_genes
        self.prism = None

        # prepare all the data
        self._prepare_data()

    def _prepare_data(self):

        # Read the sample-prior table. We need to know what samples we are using
        with Timer("Reading sample-prior configuration..."):
            self.samples, self.sample2prior_dict, self.prior2sample_dict = io.read_priors_table(self.priors_table_file)
            self.n_samples = len(self.samples)

            if self.ppi_mode == 'sample':
                self.samples_ppi, self.sample2ppi_dict, self.ppi2sample_dict = io.read_priors_table(self.ppi_table_file, sample_col='sample', prior_col='prior')
                # TODO: add check that samples ppi == samples motif

            # prepare universe of names in the priors. We won't be reading all of them
            # first, because we might want to use too many motif priors

            # from motif data
            (
                self.priors_tfs,
                self.priors_genes,
            ) = io.read_motif_universe(
                self.sample2prior_dict, mode=self.mode_priors
            )

            # from ppi data
            # if ppi table file is specified
            with Timer("Loading PPI data ..."):
                if self.ppi_mode == 'sample':
                    (
                        self.ppi_tfs,
                    ) = io.read_ppi_universe(
                        self.sample2ppi_dict, mode=self.mode_priors
                    )
                    self.ppi_data = None
                else:
                    # read ppi
                    self.ppi_data, self.ppi_tfs = io.read_ppi(self.ppi_file)

        with Timer("Reading expression data..."):
            # Read expression
            self.expression_data, self.expression_genes = io.prepare_expression(
                self.expression_file, samples=self.samples
            )

        # depending on the strategy for
        if self.mode_process == 'intersection':
            self.universe_genes = sorted(list(set(self.expression_genes).intersection(set(self.priors_genes))))
            self.universe_tfs = sorted(list(set(self.ppi_tfs).intersection(set(self.priors_tfs))))
        else:
            sys.exit('Only intersection is an available modeProcess for the moment')

        # Auxiliary dicts
        self.gene2idx = {x: i for i, x in enumerate(self.universe_genes)}
        self.tf2idx = {x: i for i, x in enumerate(self.universe_tfs)}

        # sort the gene expression and ppi data
        self.expression_data = self.expression_data.loc[self.universe_genes, self.samples]

        if self.ppi_mode == 'motif':
            self.ppi_data = self.ppi_data.loc[self.universe_tfs, self.universe_tfs]

    def run(self, keep_coexpression=False, coexpression_folder='coexpression/',
            computing_panda='cpu', alpha=0.1, precision='single', th_motifs=3,
            tune_delta=False, delta=0.1):

        """Run PrismGRN: estimate coexpression via Prism, then PANDA per sample.

        Args:
            keep_coexpression (bool, optional): whether to save each coexpression network
            coexpression_folder (str, optional): used if keep_coexpression is passed
            computing_panda (str, optional): computing for single sample panda. Defaults to 'cpu'.
            alpha (float, optional): PANDA's alpha parameter. Defaults to 0.1.
            precision (str, optional): 'single' or 'double'. Defaults to 'single'.
            th_motifs (int, optional): if the number of motif files is lower than the threshold, each will be loaded
            only once.
            tune_delta (bool, optional): calibrate delta via jackknife. Defaults to False.
            delta (float, optional): shrinkage weight, used if tune_delta=False. Defaults to 0.1.
        """

        prism_start = time.time()

        if precision not in ('single', 'double'):
            sys.exit('Precision %s unknonw' % str(precision))

        self.expression_data = self.expression_data.loc[self.universe_genes, :]

        # all coexpression computation is delegated to Prism
        self.prism = Prism(self.expression_data, delta=delta, tune_delta=tune_delta, precision=precision)
        coexpression_out = self.output_folder + coexpression_folder if keep_coexpression else './prism_coexpress/'
        self.coexpression = self.prism.compute(keep_output=keep_coexpression, output_folder=coexpression_out)

        if not os.path.exists(self.output_folder + 'single_panda/'):
            os.makedirs(self.output_folder + 'single_panda/')

        if th_motifs > len(self.prior2sample_dict.keys()):
            for p, ss in self.prior2sample_dict.items():
                motif_data, tftoadd, genetoadd = self._get_motif(p)
                for s, sample in enumerate(ss):
                    ppi_data = self._get_ppi(sample, missing_tf=tftoadd)
                    self._run_panda_coexpression(
                        self.coexpression[sample], ppi_data, motif_data, sample,
                        computing=computing_panda, alpha=alpha, save_single=True
                    )
        else:
            for s, sample in enumerate(self.samples):
                motif_data, tftoadd, genetoadd = self._get_motif(self.sample2prior_dict[sample])
                ppi_data = self._get_ppi(sample, missing_tf=tftoadd)
                self._run_panda_coexpression(
                    self.coexpression[sample], ppi_data, motif_data, sample,
                    computing=computing_panda, alpha=alpha, save_single=True
                )

    def _save_single_panda_net(self, net, prior, sample, prefix, pivot=False):

        tab = pd.DataFrame(net, columns=self.universe_genes)
        tab['tf'] = self.universe_tfs

        if pivot:
            tab.set_index('tf').to_csv(prefix + sample + '.csv')
        else:
            tab = pd.melt(tab, id_vars='tf', value_vars=tab.columns, var_name='gene', value_name='force')
            tab['motif'] = prior.flatten(order='F')
            tab.to_csv(prefix + sample + '.txt', sep='\t', index=False, columns=['tf', 'gene', 'motif', 'force'])

    def _get_motif(self, motif_fn):
        motif_data, tftoadd, genetoadd = io.read_motif(motif_fn, tf_names=list(self.universe_tfs),
                                                         gene_names=list(self.universe_genes), pivot=True)
        return (motif_data, tftoadd, genetoadd)

    def _get_ppi(self, sample, missing_tf=None):
        if (self.ppi_mode == 'sample'):
            data = io.read_ppi(self.sample2ppi_dict[sample], self.universe_tfs)
        else:
            data = self.ppi_data
            if missing_tf:
                data.loc[missing_tf, :] = 0
                data.loc[:, missing_tf] = 0

        return (data)

    def _run_panda_coexpression(self, net, ppi, motif, sample, computing='cpu', alpha=0.1, save_single=False):

        panda_loop_time = time.time()

        if (len(ppi.index) != np.sum(ppi.index == motif.index)):
            sys.exit('PPI and motif tfs are not matching. DEBUG!')
        if (len(net.index) != np.sum(motif.columns == net.index)):
            sys.exit('coexpression and motif genes are not matching. DEBUG!')
        final = calc.compute_panda(
            calc.normalize_network(net.values),
            calc.normalize_network(ppi.values),
            calc.normalize_network(motif.astype(float).values),
            computing=computing,
            alpha=alpha,
        )
        print("Running panda took: %.2f seconds!" % (time.time() - panda_loop_time))

        if save_single:
            self._save_single_panda_net(final, motif.values, sample, prefix=self.output_folder + 'single_panda/', pivot=False)
        return (final)

    def _normalize_network(self, net):
        """Normalize a network the way Panda does. Delegates to netZooPy's
        Panda implementation via a throwaway instance-free call pattern."""
        return Panda._normalize_network(self, net)


def prism_coexpress(expression_file, delta=0.1, tune_delta=True, precision='single',
                     keep_output=False, output_folder='./prism_coexpress/'):
    """One-line convenience wrapper around Prism(...).compute(...).

    Returns
    -------
    dict[str, pandas.DataFrame]
        Sample name -> (genes x genes) coexpression matrix.
    """
    model = Prism(expression_file, delta=delta, tune_delta=tune_delta, precision=precision)
    return model.compute(keep_output=keep_output, output_folder=output_folder)


def prism_multiomic_coexpress(expression_file, delta=0.1, tune_delta=True, precision='single',
                               keep_output=False, output_folder='./prism_multiomic_coexpress/'):
    """One-line convenience wrapper around PrismMultiomic(...).compute(...).

    Returns
    -------
    dict[str, pandas.DataFrame]
        Sample name -> (genes x genes) coexpression matrix, estimated
        after a nonparanormal transform of the input data.
    """
    model = PrismMultiomic(expression_file, delta=delta, tune_delta=tune_delta, precision=precision)
    return model.compute(keep_output=keep_output, output_folder=output_folder)


def prism_GRN(expression_file, priors_table_file, ppi_table_file=None, ppi_file=None,
              mode_process="union", mode_priors="union", prior_tf_col=0, prior_gene_col=1,
              output_folder='./prism/', **run_kwargs):
    """Thin wrapper: coexpression estimation (via Prism) + PANDA GRN inference.

    Equivalent to instantiating PrismGRN and calling .run(**run_kwargs).
    """
    model = PrismGRN(
        expression_file, priors_table_file, ppi_table_file=ppi_table_file, ppi_file=ppi_file,
        mode_process=mode_process, mode_priors=mode_priors,
        prior_tf_col=prior_tf_col, prior_gene_col=prior_gene_col, output_folder=output_folder
    )
    model.run(**run_kwargs)
    return model


def simulate_prism_data(
    n_genes=50,
    n_tfs=10,
    n_samples=10,
    n_groups=2,
    group_proportions=None,
    frac_diff=0.1,
    ppi_density=0.4,
    motif_density=0.3,
    ppi_sharing=0.25,
    seed=42,
    output_folder='sim_data/',
):
    """Simulate synthetic data for Prism benchmarking.

    Generates gene expression, motif priors, PPI, and a priors table
    compatible with the Prism input format. Samples are split across
    groups according to group_proportions, or evenly if not specified.

    Parameters
    ----------
    n_genes : int
        Number of target genes.
    n_tfs : int
        Number of transcription factors.
    n_samples : int
        Total number of samples.
    n_groups : int
        Number of sample groups, each with a distinct motif prior.
    group_proportions : list of float or None
        Proportion of samples assigned to each group. Must sum to 1 and
        have length equal to n_groups. If None, samples are split evenly.
        The last group absorbs any rounding remainder.
    frac_diff : float
        Fraction of TF-gene edges that differ between the base motif
        and each group motif.
    ppi_density : float
        Probability of an edge existing in the random PPI network.
    motif_density : float
        Probability of a TF regulating a gene in the base motif.
    ppi_sharing : float
        Fraction of exclusive targets shared between interacting TF pairs
        (makes the motif consistent with PPI structure).
    seed : int
        Random seed for reproducibility.
    output_folder : str
        Directory where simulated files are saved.

    Returns
    -------
    genes : list of str
    tfs : list of str
    samples : list of str
    group_motifs : list of np.ndarray
        One motif array (n_tfs x n_genes) per group.
    sample2group : dict
        Maps each sample name to its group index (0-based).
    """
    np.random.seed(seed)
    os.makedirs(output_folder, exist_ok=True)

    if group_proportions is None:
        n_samples_per_group = [n_samples // n_groups] * n_groups
        n_samples_per_group[-1] += n_samples - sum(n_samples_per_group)
    else:
        if len(group_proportions) != n_groups:
            raise ValueError(
                f"group_proportions has {len(group_proportions)} elements "
                f"but n_groups={n_groups}."
            )
        if not np.isclose(sum(group_proportions), 1.0):
            raise ValueError("group_proportions must sum to 1.")
        n_samples_per_group = [int(p * n_samples) for p in group_proportions]
        n_samples_per_group[-1] += n_samples - sum(n_samples_per_group)

    genes = [f'gene{i}' for i in range(n_genes)]
    tfs = [f'tf{i}' for i in range(n_tfs)]
    samples = [f'sample{i}' for i in range(n_samples)]

    ppi = (np.random.rand(n_tfs, n_tfs) < ppi_density).astype(float)
    ppi = np.maximum(ppi, ppi.T)
    np.fill_diagonal(ppi, 1)

    base_motif = (np.random.rand(n_tfs, n_genes) < motif_density).astype(float)
    for i in range(n_tfs):
        for j in range(i + 1, n_tfs):
            if ppi[i, j] == 1:
                only_i = np.where((base_motif[i] == 1) & (base_motif[j] == 0))[0]
                only_j = np.where((base_motif[j] == 1) & (base_motif[i] == 0))[0]
                n_share_i = int(ppi_sharing * len(only_i))
                n_share_j = int(ppi_sharing * len(only_j))
                if n_share_i > 0:
                    base_motif[j, np.random.choice(only_i, n_share_i, replace=False)] = 1
                if n_share_j > 0:
                    base_motif[i, np.random.choice(only_j, n_share_j, replace=False)] = 1

    n_diff = int(frac_diff * n_tfs * n_genes)

    def _make_group_motif(base, n_diff):
        motif = base.copy().flatten()
        flip_idx = np.random.choice(len(motif), n_diff, replace=False)
        motif[flip_idx] = 1 - motif[flip_idx]
        return motif.reshape(base.shape)

    group_motifs = [_make_group_motif(base_motif, n_diff) for _ in range(n_groups)]

    ppi_cov = ppi + np.eye(n_tfs) * 0.1
    ppi_cov = ppi_cov / ppi_cov.max()
    tf_activities = np.random.multivariate_normal(
        mean=np.zeros(n_tfs), cov=ppi_cov, size=n_samples
    ).T

    expr = np.zeros((n_genes, n_samples))
    sample2group = {}
    start = 0
    for g, n_g in enumerate(n_samples_per_group):
        idx = slice(start, start + n_g)
        tf_activities[:, idx] += np.random.randn(n_tfs, 1) * 0.5
        expr[:, idx] = (
            group_motifs[g].T @ tf_activities[:, idx]
            + np.random.randn(n_genes, n_g) * 0.5
        )
        for s in samples[start: start + n_g]:
            sample2group[s] = g
        start += n_g

    expr_df = pd.DataFrame(expr, index=genes, columns=samples)
    expr_df.index.name = 'gene'
    expr_df.to_csv(output_folder + 'expression.txt', sep='\t')

    def _save_motif(motif, path):
        rows = [
            [tf, gene, 1]
            for i, tf in enumerate(tfs)
            for j, gene in enumerate(genes)
            if motif[i, j] == 1
        ]
        pd.DataFrame(rows).to_csv(path, sep='\t', index=False, header=False)

    motif_paths = []
    for g, motif in enumerate(group_motifs):
        path = output_folder + f'motif_group{g + 1}.txt'
        _save_motif(motif, path)
        motif_paths.append(path)

    ppi_rows = [
        [tfs[i], tfs[j], ppi[i, j]]
        for i in range(n_tfs)
        for j in range(n_tfs)
        if ppi[i, j] > 0
    ]
    pd.DataFrame(ppi_rows).to_csv(
        output_folder + 'ppi.txt', sep='\t', index=False, header=False
    )

    priors_rows = [
        [s, motif_paths[sample2group[s]]] for s in samples
    ]
    pd.DataFrame(priors_rows, columns=['sample', 'prior']).to_csv(
        output_folder + 'priors_table.txt', sep=',', index=False
    )

    return genes, tfs, samples, group_motifs, sample2group


def evaluate_networks(
    output_folder,
    tfs,
    genes,
    sample2group,
    group_motifs,
):
    """Evaluate Prism output networks against known ground-truth motifs.

    Generalizes evaluation to any number of groups/motifs by using a
    sample2group mapping (mirrors the sample2prior_dict logic in Prism).
    Handles genes absent from the PANDA output (e.g., genes not regulated
    by any TF in the motif) by filling missing entries with 0.

    Parameters
    ----------
    output_folder : str
        Path to the Prism output folder (expects single_panda/ subfolder).
    tfs : list of str
        Ordered list of TF names used in the simulation.
    genes : list of str
        Ordered list of gene names used in the simulation.
    sample2group : dict
        Maps each sample name to its group index (0-based), e.g.
        {'sample0': 0, 'sample1': 0, 'sample2': 1, ...}.
    group_motifs : list of np.ndarray
        Ground-truth motif array (n_tfs x n_genes) for each group,
        indexed to match sample2group values.

    Returns
    -------
    aurocs : list of float
        Per-sample AUROC scores (samples with undefined AUROC are skipped).
    """
    from sklearn.metrics import roc_auc_score

    aurocs = []
    files = sorted(glob.glob(output_folder + 'single_panda/*.txt'))

    for f in files:
        sample = os.path.splitext(os.path.basename(f))[0]
        if sample not in sample2group:
            print(f"Sample {sample}: not found in sample2group, skipping.")
            continue

        net = pd.read_csv(f, sep='\t')
        true_motif = group_motifs[sample2group[sample]]

        net_pivot = net.pivot_table(index='tf', columns='gene', values='force')
        net_pivot = net_pivot.reindex(index=tfs, columns=genes, fill_value=0)

        true_flat = true_motif.flatten()
        pred_flat = net_pivot.values.flatten()

        if len(np.unique(true_flat)) < 2:
            print(f"Sample {sample}: skipped (ground truth is all-zero or all-one).")
            continue

        auroc = roc_auc_score(true_flat, pred_flat)
        aurocs.append(auroc)
        print(f"Sample {sample} AUROC: {auroc:.3f}")

    if aurocs:
        print(f"\nMean AUROC: {np.mean(aurocs):.3f}")
    else:
        print("\nNo samples could be evaluated.")

    return aurocs
