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


class Prism(Panda):
    """
    Personalized Regulation Inference via Sample-specific Motifs
        1. Reading in input data (expression, motif prior table, TF PPI data)
        2. Preparing motif prior universe
        3. Estimating sample-specific coexpression with lioness
        4. Running PANDA with a different prior for each samples
    
    Warning: if you are familiar with the other netzoopy functions, this one is slightly different. 
    We have separated the reading and preprocessing steps from those for computation in a more 
    OOP-friendly fashion.


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
        ppi_file = None,
        mode_process="union",
        mode_priors="union",
        prior_tf_col=0,
        prior_gene_col=1,
        output_folder='./prism/'
    ):
        """Intialize instance of Panda class and load data."""

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
        self.prior_gene_col=prior_gene_col
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


        # prepare all the data
        self._prepare_data()

    ########################
    ### METHODS ############
    ########################
    def _prepare_data(self):

        # Read the sample-prior table. We need to know what samples we are using
        with Timer("Reading sample-prior configuration..."):
            self.samples, self.sample2prior_dict, self.prior2sample_dict = io.read_priors_table(self.priors_table_file)
            self.n_samples = len(self.samples)

            if self.ppi_mode=='sample':
                self.samples_ppi, self.sample2ppi_dict, self.ppi2sample_dict = io.read_priors_table(self.ppi_table_file, sample_col = 'sample', prior_col = 'prior')
                #TODO: add check that samples ppi == samples motif

            # prepare universe of names in the priors. We won't be reading all of them 
            # first, because we might want to use too many motif priors

            # from motif data
            (
                self.priors_tfs,
                self.priors_genes,
            ) = io.read_motif_universe(
                self.sample2prior_dict, mode=self.mode_priors
            )
            
            #from ppi data
            # if ppi table file is specified
            with Timer("Loading PPI data ..."):
                if self.ppi_mode=='sample':
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
        if self.mode_process=='intersection':
            self.universe_genes = sorted(list(set(self.expression_genes).intersection(set(self.priors_genes))))
            self.universe_tfs = sorted(list(set(self.ppi_tfs).intersection(set(self.priors_tfs))))
        else:
            sys.exit('Only intersection is an available modeProcess for the moment')

        # Auxiliary dicts
        self.gene2idx = {x: i for i, x in enumerate(self.universe_genes)}
        self.tf2idx = {x: i for i, x in enumerate(self.universe_tfs)}

        # sort the gene expression and ppi data
        self.expression_data = self.expression_data.loc[self.universe_genes,self.samples]

        if self.ppi_mode=='motif':
            self.ppi_data = self.ppi_data.loc[self.universe_tfs,self.universe_tfs]
    
    def run_prism(self, keep_coexpression = False,save_memory = False, online_coexpression = False, coexpression_folder = 'coexpression/', computing_lioness = 'cpu', computing_panda = 'cpu', cores = 1, alpha = 0.1 , precision = 'single', th_motifs = 3, tune_delta=False, delta=0.1):
        
        """Prism algorithm

        Args:
            keep_coexpression (bool, optional): whether to save each coexpression network
            save_memory (bool, optional): whether to save the coexpression with the gene names
            online_coexpression (bool, optional): if true coexpression is computed with a closed form
            coexpression_folder (str, optional): used if keep_coexpression is passed
            computing_lioness (str, optional): computing for coexpression lioness. Defaults to 'cpu'.
            computing_panda (str, optional): computing for single sample panda. Defaults to 'cpu'.
            cores (int, optional): cores. Defaults to 1.
            alpha (float, optional): _description_. Defaults to 0.1.
            th_motifs (int, optional): if the number of motif files is lower than the threshold, each will be loaded
            only once.
        """

        prism_start = time.time()
        # first let's reorder the expression data
        
        if precision=='single':
            atype = 'float32'
        elif precision=='double':
            atype = 'float64'
        else: 
            sys.exit('Precision %s unknonw' %str(precision))
        
        # let's sort the expression and ppi data

        self.expression_data = self.expression_data.loc[self.universe_genes,:].astype(atype)
        #correlation_complete = self.expression_data.T.corr()
        # we automatically multiply the correlation with the number of samples
        #correlation_complete = correlation_complete * self.get_n_matrix(self.expression_data)
        self.n_matrix = self.get_n_matrix(self.expression_data)
        
        # consider removing this
        # scale expression data to make mean = 0 and sd = 1
        # self.expression_data_scaled = (self.expression_data - self.expression_data.mean(axis = 1))/self.expression_data.std(ddof=1, axis = 1)
    
        # Center expression data to make mean = 0
        # let's remove this from here, and keep it only inside the prism computation
        #self.expression_data_centered = (self.expression_data - np.mean(self.expression_data.values,axis = 1, keepdims=True))

        self.expression_mean = np.nanmean(self.expression_data.values,axis = 1, keepdims=True)
        self.covariance_matrix = self.expression_data.T.cov().values

        if tune_delta:
            self.delta = estimate_delta_jackknife(self.expression_data)
        else:
            self.delta = delta

        
        if th_motifs>len(self.prior2sample_dict.keys()):
            for p,ss in self.prior2sample_dict.items():
                # read the motif data and sort it
                motif_data, tftoadd, genetoadd = self._get_motif(p)
                for s,sample in enumerate(ss):
                    sample_start = time.time()
                    ppi_data = self._get_ppi(sample, missing_tf = tftoadd)
                    # first run lioness on coexpression
                    self._prism_loop(ppi_data, motif_data, sample, keep_coexpression=keep_coexpression, save_memory=save_memory, computing_lioness=computing_lioness, computing_panda=computing_panda, alpha = alpha, coexpression_folder=coexpression_folder, delta = delta, tune_delta=tune_delta)

        else:
            # Now for each sample we compute the lioness network from correlations and 
            # the panda using the motif and ppi tables
            for s,sample in enumerate(self.samples):
                sample_start = time.time()
                # first run lioness on coexpression
                motif_data, tftoadd, genetoadd = self._get_motif(self.sample2prior_dict[sample])
                ppi_data = self._get_ppi(sample, missing_tf = tftoadd)
                self._prism_loop(ppi_data, motif_data, sample, keep_coexpression=keep_coexpression, save_memory=save_memory, computing_lioness=computing_lioness, computing_panda=computing_panda, alpha = alpha, coexpression_folder=coexpression_folder, delta = delta, tune_delta=tune_delta)


    def _prism_loop(self, ppi_data, motif_data, sample, keep_coexpression = False, save_memory = True, online_coexpression = False, computing_lioness = 'cpu', coexpression_folder = './coexpression/' , computing_panda = 'cpu', alpha = 0.1, delta=0.3,tune_delta=False):
        """Runs prism on one sample. For now all samples are saved separately.

        Args:
            correlation_complete (_type_): _description_
            ppi_data (_type_): _description_
            motif_data (_type_): _description_
            sample (_type_): _description_
            keep_coexpression (bool, optional): _description_. Defaults to False.
            save_memory (bool, optional): _description_. Defaults to True.
            online_coexpression (bool, optional): _description_. Defaults to False.
            computing_lioness (str, optional): _description_. Defaults to 'cpu'.
            coexpression_folder (str, optional): _description_. Defaults to './coexpression/'.
            computing_panda (str, optional): _description_. Defaults to 'cpu'.
            alpha (float, optional): _description_. Defaults to 0.1.
        """
        if keep_coexpression:
            if not os.path.exists(self.output_folder+coexpression_folder):
                os.makedirs(self.output_folder+coexpression_folder)
        
        if not os.path.exists(self.output_folder+'single_panda/'):
            os.makedirs(self.output_folder+'single_panda/')
        sample_lioness = self._run_lioness_coexpression(sample, keep_coexpression = keep_coexpression, save_memory = save_memory, online_coexpression = online_coexpression, computing = computing_lioness, coexpression_folder = coexpression_folder, delta = delta, tune_delta = tune_delta)

        final_panda= self._run_panda_coexpression(sample_lioness,ppi_data, motif_data, sample, computing = computing_panda, alpha = alpha, save_single=True)
        #return(final_panda)

    def _save_single_panda_net(self, net, prior, sample, prefix, pivot = False):

        tab = pd.DataFrame(net, columns = self.universe_genes )
        tab['tf'] = self.universe_tfs

        if pivot:
            tab.set_index('tf').to_csv(prefix+sample+'.csv')
        else:
            tab = pd.melt(tab, id_vars='tf', value_vars=tab.columns,var_name='gene', value_name='force')
            tab['motif'] = prior.flatten(order = 'F')
            tab.to_csv(prefix+sample+'.txt', sep = '\t', index = False, columns = ['tf', 'gene','motif','force'])

    def _get_motif(self, motif_fn):
        motif_data,tftoadd, genetoadd = io.read_motif(motif_fn, tf_names = list(self.universe_tfs), gene_names = list(self.universe_genes), pivot = True)
        return(motif_data, tftoadd,genetoadd)

    def _get_ppi(self, sample, missing_tf = None):
        if (self.ppi_mode == 'sample'):
            data = io.read_ppi(self.sample2ppi_dict[sample], self.universe_tfs)
        else:
            data = self.ppi_data
            # if there are missing tf, the ppi is all null and 
            if missing_tf:
                data.loc[missing_tf,:]=0
                data.loc[:,missing_tf]=0
                #data.loc[missing_tf,missing_tf] = np.eye(len(missing_tf))

        return(data)

    def _run_lioness_coexpression(self, sample, keep_coexpression = False,save_memory = True, online_coexpression = False, computing = 'cpu', cores = 1, coexpression_folder = 'coexpression/', delta = 0.3, tune_delta = False):
        
        touse = list(set(self.samples).difference(set([sample])))
        names = self.expression_data.index.tolist()
                 
        #correlation_matrix = self.expression_data.loc[:, touse].T.corr().values
        # Compute covariance matrix from the rest of the data, leaving out sample
        # covariance_matrix = self.expression_data.loc[:, touse].T.cov().values
        
        #correlation_matrix = self.expression_data.loc[:, touse].T.corr().values
        
        # Compute covariance matrix from the rest of the data, leaving out sample
        # covariance_matrix = self.expression_data.loc[:, touse].T.cov().values
        
        # For consistency with R, we are using the N panda_all - (N-1) panda_all_but_q
        # coexpression has been already multiplied by N all
        # we no longer need coexpression
        #lioness_network = coexpression - (
        #        (self.get_n_matrix(self.expression_data.loc[:, touse])) * correlation_matrix
        #)
        
        # Compute sample-specific covariance matrix
        sscov = self.delta * np.outer((self.expression_data-self.expression_mean).loc[:, sample], (self.expression_data-self.expression_mean).loc[:, sample]) + (1-self.delta) * self.covariance_matrix

        # we no longer need coexpression
        #lioness_network = coexpression - (
        #        (self.get_n_matrix(self.expression_data.loc[:, touse])) * correlation_matrix
        #)

        # Compute sample-specific coexpression matrix from the sample-specific covariance matrix
        
        sscov = np.array(sscov)
        diag = np.sqrt(np.diag(np.diag(sscov)))

        # Replace 0 diagonals by 1, so that the diagonal matrix can be inverted
        diag = np.array(diag)
        indices = np.where(np.diag(diag) == 0)[0]
        for i in indices:
            diag[i,i] = 1
            
        sds = np.linalg.inv(diag)
        lioness_network = sds @ sscov @ sds

        nmatrix = self.n_matrix - self.get_n_matrix(self.expression_data.loc[:, touse])

        lioness_network = pd.DataFrame(data = np.multiply(nmatrix, lioness_network), index = names, columns=names)
       

        if (keep_coexpression):
            cfolder = self.output_folder+coexpression_folder
            if not os.path.exists(cfolder):
                os.makedirs(cfolder)
            path = cfolder+'coexpression_'+sample
            path_genename = cfolder+'genenames_'+sample
            if (save_memory):
                #if self.save_fmt == "txt":
                #np.savetxt(path+'.txt', coexp)
                #elif self.save_fmt == "npy":
                np.save(path+'.npy', lioness_network.values)
                # write the gene names
                with open(path_genename+'.txt', 'w') as fp:
                    for item in names:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                #elif self.save_fmt == "mat":
                #    from scipy.io import savemat
                #    savemat(path, {"SSCoexp": coexp})
            else:
                pd.DataFrame(data = lioness_network.values, columns=names, index = names).to_csv(cfolder+'coexpression_'+sample+'.txt', sep = ' ')
        
        return(lioness_network)



    def _run_panda_coexpression(self, net, ppi, motif, sample, computing = 'cpu', alpha = 0.1, save_single = False):
        
        panda_loop_time = time.time()
        
        #panda works with all normalised networks
        if (len(ppi.index)!=np.sum(ppi.index==motif.index)):
            sys.exit('PPI and motif tfs are not matching. DEBUG!')
        if (len(net.index)!=np.sum(motif.columns==net.index)):
            sys.exit('coexpression and motif genes are not matching. DEBUG!')
        final = calc.compute_panda(
            self._normalize_network(net.values),
            self._normalize_network(ppi.values),
            self._normalize_network(motif.astype(float).values),
            computing=computing,
            alpha=alpha,
        )
        print("Running panda took: %.2f seconds!" % (time.time() - panda_loop_time))

        if save_single:
            self._save_single_panda_net(final, motif.values, sample, prefix = self.output_folder+'single_panda/', pivot = False)
        return(final)

    def get_n_matrix(self,df):
        # This should be outside of the class
        """Get number of samples for each correlation value

        Args:
            df (pd.DataFrame): expression with nan values
        """
        
        N = len(df.columns)
        nn = N-df.isna().sum(axis = 1).values[:,np.newaxis]
        nr = np.repeat(nn,len(nn), axis = 1)
        return(np.minimum(nr,nr.T))

def estimate_delta_jackknife(expression_data):
    """Efficient closed-form jackknife estimate of delta (eq. 2.11 in the paper).

    Parameters
    ----------
    expression_data : pandas.DataFrame or np.ndarray, shape (g, n)
        Genes (rows) x samples (columns). Should be the same data used to
        compute self.covariance_matrix (i.e., already restricted to
        self.universe_genes).

    Returns
    -------
    float
        Estimated delta.
    """
    X = np.asarray(expression_data, dtype='float64')
    g, n = X.shape

    # Population-level diagonal of S: per-gene variance, ddof=1
    s_kk = X.var(axis=1, ddof=1)  # shape (g,)

    # Closed-form leave-one-sample-out variance per gene, vectorized
    s1 = X.sum(axis=1, keepdims=True)          # (g, 1)
    s2 = (X ** 2).sum(axis=1, keepdims=True)   # (g, 1)

    s1_loo = s1 - X                              # (g, n)
    s2_loo = s2 - X ** 2                         # (g, n)
    mean_loo = s1_loo / (n - 1)
    var_loo = (s2_loo - (n - 1) * mean_loo ** 2) / (n - 2)  # (g, n)

    eta = var_loo.var(axis=1, ddof=1)  # (g,) -- one eta^(k) per gene

    numerator = 2 * np.sum(s_kk ** 2)
    denominator = np.sum(eta)

    nu = 3 + numerator / denominator
    delta = 1.0 / nu
    return delta


########################
### SIMULATION #########
########################

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

    # SAMPLE COUNTS PER GROUP
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

    genes   = [f'gene{i}' for i in range(n_genes)]
    tfs     = [f'tf{i}'   for i in range(n_tfs)]
    samples = [f'sample{i}' for i in range(n_samples)]

    # PPI: random symmetric network
    ppi = (np.random.rand(n_tfs, n_tfs) < ppi_density).astype(float)
    ppi = np.maximum(ppi, ppi.T)
    np.fill_diagonal(ppi, 1)

    # BASE MOTIF: PPI-consistent
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

    # GROUP MOTIFS: flip frac_diff edges from base
    n_diff = int(frac_diff * n_tfs * n_genes)

    def _make_group_motif(base, n_diff):
        motif = base.copy().flatten()
        flip_idx = np.random.choice(len(motif), n_diff, replace=False)
        motif[flip_idx] = 1 - motif[flip_idx]
        return motif.reshape(base.shape)

    group_motifs = [_make_group_motif(base_motif, n_diff) for _ in range(n_groups)]

    # EXPRESSION: TF activities drive gene expression
    ppi_cov = ppi + np.eye(n_tfs) * 0.1
    ppi_cov = ppi_cov / ppi_cov.max()
    tf_activities = np.random.multivariate_normal(
        mean=np.zeros(n_tfs), cov=ppi_cov, size=n_samples
    ).T  # shape: n_tfs x n_samples

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

    # SAVE: expression
    expr_df = pd.DataFrame(expr, index=genes, columns=samples)
    expr_df.index.name = 'gene'
    expr_df.to_csv(output_folder + 'expression.txt', sep='\t')

    # SAVE: motif priors (one file per group)
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

    # SAVE: PPI
    ppi_rows = [
        [tfs[i], tfs[j], ppi[i, j]]
        for i in range(n_tfs)
        for j in range(n_tfs)
        if ppi[i, j] > 0
    ]
    pd.DataFrame(ppi_rows).to_csv(
        output_folder + 'ppi.txt', sep='\t', index=False, header=False
    )

    # SAVE: priors table
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
        # Use reindex instead of .loc to handle genes absent from the PANDA output
        # (genes with no regulating TFs in the motif are missing from the network
        # file; filling with 0 treats them as unregulated, which is biologically
        # consistent)
        net_pivot = net_pivot.reindex(index=tfs, columns=genes, fill_value=0)

        true_flat = true_motif.flatten()
        pred_flat = net_pivot.values.flatten()

        # AUROC is undefined when the ground truth contains only one class
        # (e.g., all edges are 0 because no TF regulates any gene in this sample)
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
