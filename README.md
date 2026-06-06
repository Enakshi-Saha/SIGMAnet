<p align="center">
  <img src="docs/logo.png" alt="SIGMAnet logo" width="600"/>
</p>

# SIGMAnet: Sample-specific and Integrated Graphical Models for Association Networks

SIGMAnet is a Python package for estimating individual-specific 
molecular networks from omics data. The current release includes 
**SIREN** (Sample-specific Inference via Regularized 
Empirical-Bayes Networks), a method for estimating 
individual-specific partial correlation networks from 
high-dimensional multi-omics data.

## Installation

```bash
git clone https://github.com/Enakshi-Saha/SIGMAnet.git
cd SIGMAnet
pip install -e Python/
```

## Quick Start

```python
from siren.siren import Siren

# Single omics
siren_obj = Siren(data)

# Two omics (expression + methylation)
siren_obj = Siren(expression_file, methylation_file)

# Run SIREN
siren_obj.run_siren(keep_in_memory=False, 
                    output_fmt='.h5',
                    output_folder='./siren_output/')
```

## Reference

Saha, E. (2026). Individual-Specific Gaussian Graphical Models 
for Heterogeneous Populations with Application to Epigenetic 
Gene Regulation in Lung Adenocarcinoma. *arxiv*.

<!--
Future methods planned for SIGMAnet:
- ECLIPSE: Empirical Bayes Covariance Learning with Imputation 
  for Personalized Statistical Estimation (sample-specific 
  correlation with missing data handling for single-cell data)
- PRISM: Personalized Regulation Inference via Sample-specific 
  Motifs (sample-specific GRN)
- MIRAGE: Multi-omic Individual-specific RegulArized GGM Estimation (Smaug reformulated)
- Sample-specific factor analysis (X = WB + Error, where 
  Cov(X) is sample-specific)
-->
