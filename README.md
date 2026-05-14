# SIGMAnet
Sample-specific and Integrated Graphical Models for Association Networks.

At current stage three methods are hosted:

- **Sample-specific correlation** (improves BONOBO wth full empirical Bayes formulation and missing data handling capability to enable single-cell data). The name is "ECLIPSE: Empirical Bayes Covariance Learning with Imputation for Personalized Statistical Estimation". Imputation because we impute corrletaion value with the prior correlation if individual (cell-level) data is missing.
- **Sample-specific GRN** PRISM — Personalized Regulation Inference via Sample-specific Motifs
- **Sample-specific factor analysis** (X = WB + Error, where Cov(X) is sample-specific and estimated using the above method, then a factorization is done from the marginal model Cov(X) = W*W + I)
- **Sample-specific Gaussian Graphical Model** (uses OAS by Chen, 2010 to estimate a prior penalized covariance matrix and uses this to estimate posterior inverse Wishart GGM, using Sherman Morrison for scalability. The name is SIREN: Sample-specific Inference via Regularized Empirical-bayes Networks.

SIGMAnet include two plotting functions for unipartite and bipartite networks respectively.
