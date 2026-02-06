# MTX_utils
Benchmarking of metatranscriptomic differential expression using synthetic and in vitro mock communities.

If you use this code, please cite our manuscript: 

Lee et al., "Enhancing inference of differential gene expression in metatranscriptomes from human microbial communities" (Under review).

This repository contains the following modules: 

bacteria_info.py: Utility functions for working with bacterial genomes and functional annotations (mcSEED, CAZy)

benchmarking.py: Utility functions for loading differential expression results, defining benchmarking sets, and calculation/visualization of performance metrics

kallisto_data_utils.py: Functions for processing kallisto and bowtie2 outputs into counts tables, loading and summary metric calculations on counts tables, 
formatting method input for differential expression tools, and some general visualization plot functions (volcano plots, bar-swarm plots, etc)

MTX_colors.py: Color palettes used across different analyses presented in the paper. 

simulation.py: Functions for estimating and visualizing gene-level mean, variance, and negative binomial dispersion parameters

synth_data_utils.py: Utility functions for working with synthetic datasets output from MTX_synthetic (https://github.com/biobakery/MTX_synthetic, developed by Eric Franzosa)

Generally the following dependencies are required, with versions used to produce the results in Lee et al. in parentheses: 

numpy (1.26.4)

pandas (1.5.1)

scipy (1.11.4)

matplotlib (3.6.2)

seaborn (0.13.0)


