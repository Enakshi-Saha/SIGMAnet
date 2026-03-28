# SIGMAnet
Sample-specific Individualized Graphical Models for Association Networks.

At current stage three methods are hosted:

- **Sample-specific correlation** (improves BONOBO wth full empirical Bayes formulation and missing data handling capability to enable single-cell data)
- **Sample-specific factor analysis** (X = WB + Error, where Cov(X) is sample-specific and estimated using the above method, then a factorization is done from the marginal model Cov(X) = W*W + I)
- **Sample-specific Gaussian Graphical Model** (uses OAS by Chen, 2010 to estimate a prior penalized covariance matrix and uses this to estimate posterior inverse Gamma GGM, using Sherman Morrison for scalability.
