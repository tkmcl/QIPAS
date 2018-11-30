# QIPAS
### Automated workflows for QI analysis

This repo contains notebooks and code to automate a number of seismic QI (quantitative interpretation) workflows.
Workflows in place (fully/partially tested) are:
- automated attribute extraction from a seismic cube along horizons
- automated splitting of attribute maps in blocks
- automated CTD stacking within a block to determine fluid contacts
- automated DHI (direct hydrocarbon) above/below contact amplitude analysis to extract (ML-style) features that allow for the correlation of amplitude/attributes with fluid presence

Contents:
- HCM_QI_analysis.py: contains DHI analysis modules
- HCM_QI_analysis.ipynb: example to run the DHI analysis module
- HCM_QI_ML.ipynb: example to run ML on the result of the DHI analysis

<i>kudos to Yuanzhong Fan for drafting the majority of the code</i>
