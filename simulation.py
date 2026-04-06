#! /usr/bin/env python

"""
simulation.py
Utility functions for characterizing mean-variance relationships
from genomes in mock communities and MAG datasets and for 
simulating 'human community'-like metatranscriptomic profiles
from mock community datasets. 
Evan Lee
Last update: 12/16/25
"""

###============================================================================###
# Imports 
###============================================================================###

import pandas as pd
import numpy as np
import os
import re 
import warnings
import sys 
import warnings
from scipy import stats

import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import seaborn as sns 
from MTX_utils import bacteria_info, MTX_colors


###============================================================================###
# Mean variance trend helper functions  
###============================================================================###

def get_top_n_samples(genome_mtx,n_samples=6):
    genome_depths = genome_mtx.sum(axis=0) #sum across genes within samples
    sorted_genome_depths = genome_depths.sort_values(ascending=False)
    top_n_samples = sorted_genome_depths.index[:n_samples]
    top_n_depths = sorted_genome_depths[top_n_samples]
    return top_n_samples, top_n_depths

def genome_counts_mean_variance_df(genome_mtx,
                                   depth_normalize=True,
                                   log_transform=True,
                                   log_pseudocount=1,
                                   estimate_dispersions=False,
                                   sub_min_dispersion=False,
                                   min_dispersion=10**-8
                                   ):
    #If depth_normalize, scale all samples so that they have equal depths 
    # to highest depth sample
    if depth_normalize:
        print("Normalizing by depth using scaling factors in genome_counts_mean_variance_df")
        #Replace call handles 0 total genome counts (0 counts for all genes)
        # These will be left intact when scaling counts
        scaling_factors = genome_mtx.sum().max()/(genome_mtx.sum().replace(0,1))
        # scaling_factors = genome_mtx.sum().min()/(genome_mtx.sum().replace(0,1))
        
        genome_mtx = np.rint(genome_mtx*scaling_factors).astype(int)
    #Generate summary stats DF - indexed on genes, metrics as columns
    mean_var_df = pd.DataFrame(index=genome_mtx.index)
    mean_var_df['mean'] = genome_mtx.mean(axis=1)
    #Variance metrics calculation:
    mean_var_df['var'] = genome_mtx.var(axis=1)
    mean_var_df['std'] = genome_mtx.std(axis=1)
    #Log transform metrics into separate columns if applicable 
    if log_transform:
        for col in mean_var_df.columns:
            log_col = 'log_{0}'.format(col)
            mean_var_df[log_col] = np.log10(mean_var_df[col]+log_pseudocount)
    #Use NB mean-variance relationship to estimate dispersions for each gene 
    # We use the method of moments method here even though it is invalid for 
    # underdispersed counts; we'll be representing dispersion across the 
    # genome with its median dispersion (c/w Gierlinkski et al Bioinformatics 2015)
    if estimate_dispersions:
        mean_var_df['dispersion'] = (mean_var_df['var']-\
                                              mean_var_df['mean'])/\
                                                mean_var_df['mean']**2
        if sub_min_dispersion:
            #Substitute in minDisp for edge cases where MOM dispersioon is not defined 
            # or where MOM estimates are below min_dispersion
            # The value used here is based on DESeq2's min_disp value: 
            # https://github.com/thelovelab/DESeq2/blob/devel/R/core.R (line 658)
            mean_var_df.loc[(mean_var_df['mean']==0) | 
                            (mean_var_df['dispersion']<min_dispersion),
                            'dispersion'] = min_dispersion
        #Calculate log dispersion and BCV for visualization purposes 
        mean_var_df['log_dispersion'] = np.log10(mean_var_df['dispersion'])
        mean_var_df['BCV'] = np.sqrt(mean_var_df['dispersion'])
    return mean_var_df

#Bootstrapping mtx profiles 
def bootstrap_mtx_sample(genome_mtx,axis=1):
    #Bootstrap sample MTX profile samples (columns)
    if axis == 1:
        n_samples = len(genome_mtx.columns)
    #Bootstrap sample MTX profile genes (rows) - violates many assumptions about 
    # sampling wrt observations 
    elif axis == 0:
        n_samples = len(genome_mtx)
    else:
        raise ValueError("axis must be 0 (rows) or 1 (cols).")
    return genome_mtx.sample(n=n_samples,replace=True,axis=axis)

def estimate_dispersion_ci(genome_mtx,
                           depth_normalize=True,
                           n_bootstraps=1000,
                           ci=95, random_seed=0,
                           bootstrap_axis=1):
    #Estimate confidence intervals on dispersion estimates using bootstrapping 
    # of counts profiles in genome_mtx.   
    if random_seed:
        np.random.seed(random_seed)
    #Standardize library sizes and reintegerize with rint
    if depth_normalize:
        scaling_factors = genome_mtx.sum().max()/(genome_mtx.sum().replace(0,1))
        genome_mtx = np.rint(genome_mtx*scaling_factors).astype(int)
    bootstrap_df = pd.DataFrame(columns=['dispersion_med','dispersion_mean'])
    #Bootstrap sample from genome_mtx, then calculate mean/var/estimate MME dispersion
    # Then take central tendency with median/mean
    
    for i in range(n_bootstraps):
        bootstrap_sample_mtx = bootstrap_mtx_sample(genome_mtx,axis=bootstrap_axis)
        bootstrap_mean_var_metrics = genome_counts_mean_variance_df(bootstrap_sample_mtx,
                                                                    depth_normalize=False,
                                                                    log_transform=False,
                                                                    estimate_dispersions=True)
        bootstrap_df.loc[i,'dispersion_med'] = bootstrap_mean_var_metrics['dispersion'].median()
        bootstrap_df.loc[i,'dispersion_mean'] = bootstrap_mean_var_metrics['dispersion'].mean()
    #CIs by quantile:
    # First convert CI into corresponding quantiles (split ci around 0.50)
    quantile_higher = 0.5 + (ci/2)/100
    quantile_lower = 0.5 - (ci/2)/100
    disp_ci_upper = bootstrap_df['dispersion_med'].quantile(quantile_higher)
    disp_ci_lower = bootstrap_df['dispersion_med'].quantile(quantile_lower)
    return bootstrap_df, disp_ci_upper, disp_ci_lower
    
###============================================================================###
# Quasi-simulation: rarefaction helper functions  
###============================================================================###

def genome_mtx_depth_categories(genome_mtx,metadata,
                                group_col='Mixture',
                                replicate_col='Replicate'):
    """
    Returns a DataFrame indexed on samples containing 1) genome-level depths 
    and 2) group and replicate labels from metadata. 
    
    :param genome_mtx: Genes x samples DataFrame corresponding to samples in metadata
    :param metadata: Sample indexed DataFrame, must contain labels specified by 
        group_col and replicate_col. 
    :param group_col: Label in metadata which will be propagated. Contains 
        information on groups of samples which are replicates.   
    :param replicate_col: Label in metadata which will be propagated. Contains 
        information on biological replicate numbers within each group
    """
    
    depth_df = genome_mtx.sum(axis=0).to_frame().rename(columns={0:'Depth'})
    if group_col not in metadata.columns or \
        replicate_col not in metadata.columns: 
        raise ValueError("Provided metadata need 'Replicate' and " \
                        "group_col ('{0}') labels.".format(group_col))
    depth_df[group_col]  = metadata[group_col]
    depth_df[replicate_col]  = metadata[replicate_col]
    return depth_df

def select_mock_group_by_target_depth(target_depth,
                                      depth_df,
                                      genome_mtx,
                                      group_col='Mixture'):
    """
    For a given target depth, randomly select a MTX sample profile from the 
    group of samples in mock_depths_groups with the lowest depth that is greater 
    than the target depth. 
    
    :param target_depth: Target depth to rarefy mock community reads to
    :param depth_df: Samples x metadata label, must contain labels 'Depth' 
        (per sample reads for genome of interest) and group_col (e.g. 'Mixture')
        which will be used to group samples for assessing group mean depth. 
    :param group_col: Metadata label in depth_df to group samples by.  

    :return group_mtx: Genes x samples
    """
    #Average depth per group ('mean' agg)
    depths_gb = depth_df[['Depth',group_col]].groupby(group_col).agg('mean',numeric_only=True)
    #Number of replicates per group ('count' agg)
    depths_gb['Replicate'] = depth_df[['Mixture','Replicate']].groupby('Mixture').agg('count')
    #Sort depth groups by Depth in case not already done 
    depths_gb = depths_gb.sort_values('Depth',ascending=False)
    #Choose group level with lowest depth that is greater than target depth
    above_target_group_depths = depths_gb[depths_gb['Depth']>=target_depth]
    if len(above_target_group_depths) == 0:
        warnings.warn("Target depth {0} is higher than depths all group depths. " \
                      "Sample will be taken from group with highest mean depth "
                      "and no rarefaction will be applied.".format(target_depth))
        #Get highest avg depth group:
        mock_group_to_sample = depths_gb['Depth'].idxmax()
    else:
        mock_group_to_sample = above_target_group_depths['Depth'].idxmin()
    group_samples = depth_df[depth_df[group_col]==mock_group_to_sample].index

    group_mtx = genome_mtx.loc[:,group_samples]
    return group_mtx

def select_mock_replicates_by_target_depth(target_depth,
                                            depth_df,
                                            genome_mtx,
                                            replicate_col='Replicate'):
    """
    For a given target depth, select a set of replicates which have
    genome depths (in depth_df) greater than target depth and have the 
    lowest depths that meet this criteria.  
    
    :param target_depth: Target depth to rarefy mock community reads to
    :param depth_df: Samples x metadata label, must contain labels 'Depth' 
        (per sample reads for genome of interest) and replicate_col 
        (e.g. 'Replicate'). For each unique value in replicate_col, 
        one sample will be included in the return DataFrame. 
    :param replicate_col: Metadata label containing replicate information.
    """
    #Subset depths to samples above target_depth 
    above_target_depth_df = depth_df[depth_df['Depth'] >= target_depth]
    #Initialize return DataFrame
    replicates_mtx = pd.DataFrame(index=genome_mtx.index)
    #For each biological replicate number, identify the sample with lowest 
    # depth that is above target_depth 
    for replicate in above_target_depth_df[replicate_col].unique():
        replicate_depths = above_target_depth_df[
                            above_target_depth_df[replicate_col]==replicate]
        try: 
            replicate_sample = replicate_depths['Depth'].idxmin()
        except: 
            raise ValueError("Specified target_depth is higher than all samples from.\n" \
         " replicate {0}. Cannot rarefy samples from this replicate to produce target depth"\
            .format(replicate))
        #Add replicate counts profile to return 
        replicates_mtx = pd.concat((replicates_mtx,
                                    genome_mtx[replicate_sample]),
                                   axis=1)
    return replicates_mtx


def select_random_sample_mtx(samples_mtx,
                            random_seed=0):
    """
    Given a set of sample MTX profiles, randomly choose one and return
    associated sample identifier and MTX profile (genes)
    
    :param samples_mtx: Genes x samples DataFrame of counts to randomly select
        sample_mtx from. 
    :param random_seed: Random seed for reproducible sampling. Note that this function
        generally should not be seeded if iteratively calling or it will always return the 
        same sample. Parent functions (containing iteration) should be seeded. 

    :return choice_sample: Sample identifier randomly chosen from group.
    :return choice_counts: MTX counts for choice_sample from genome_mtx
    """
    if random_seed:
        np.random.seed(random_seed)
    choice_sample = np.random.choice(samples_mtx.columns,size=1)[0]
    choice_counts = samples_mtx[choice_sample]
    return choice_sample, choice_counts
    
def generate_mock_profiles_to_rarefy(target_depths,genome_mtx,metadata,
                                     group_col='Mixture',
                                     replicate_col='Replicate',
                                     choice_method='replicate',
                                     random_seed=0
                                     ):
    """
    Given a set of target rarefaction depths (target_depths), return: 
    1) a genes x samples DataFrame consisting of mock community transcriptomes
    to rarefy. 
    2) an array of rarefaction depths; these will be equal to target depths 
    except in cases where chosen sample depths are less than target depth. 
    In these cases, the chosen sample depths will be substituted. 

    :param target_depths: array-like of int, required. Target rarefaction 
        depths for mock communities. 
    :param genome_mtx: Genes x samples DataFrame, required. Sample gene expression
        profiles which will be used as rarefaction inputs. 
    :param metadata: Samples x metadata DataFrame, required. Must contain 
        group_col and replicate_col.
    :param group_col: Label in metadata. If choice_method is 'group', 
        for each target depth, samples to rarefy are chosen from the 
        group in metadata with the minimum average depth that is higher 
        than the target depth. 
    :param replicate_col: Label in metadata. If choice method is' 'replicate',
        for each target depth, samples to rarefy are chosen from samples 
        with the minimum depth that is higher than the target such that 
        each unique value of replicate is represented in the samples to 
        choose from. 
    :param choice_method: {'replicate,'group'}, default 'replicate'. 
        'replicate': Utilize replicate metadata to construct a set of 
        mock communities representing all biological replicates, all of which 
        are the lowest depth samples for that replicate to meet the 
        depth > target_depth requirement. 
        'group': Use group metadata to average genome depths per group. Select
        mock communities from the group with the lowest average depth that 
        meets the depth > target_depth requirement. Reduce target rarefaction 
        depths to match sample depth in cases where sample depth <= target. 
    
    :return mock_mtx_to_rarefy: Genes x samples DataFrame containing samples 
        to use as input for rarefaction. Sample identifiers have been 
        deduplicated and do not correspond to original genome_mtx input.  
    :return rarefaction_depths: array-like of ints. These are the same
        as target_depths, unless choice method is group and a sample
        with lower depth than target is chosen; in this case, the chosen
        sample's depth will be substituted in rarefaction_depth. 
    """
    #If random_seed provided, use for reproducibile sampling
    if random_seed:
        np.random.seed(random_seed)
    #Generate depth_df, contains relevant metadata (group_col and replicate_col)
    # and aggregated sample genome depths (sum of all gene counts) per sample
    depth_df = genome_mtx_depth_categories(genome_mtx,metadata,
                                            group_col=group_col,
                                            replicate_col=replicate_col)
    #Generate DataFrame which will store corresponding counts profiles 
    # to use as input for rarefaction (one profile for each target depth)
    mock_mtx_to_rarefy = pd.DataFrame(index=genome_mtx.index)
    rarefaction_depths = np.zeros(len(target_depths))
    for i,target_depth in enumerate(target_depths):
        #For group sample selection, depths will be averaged within each group
        # specified by group_col. The group which has the minimum average depth 
        # greater than target depth will be used and one sample will randomly be chosen. 
        if choice_method == 'group':
            group_mtx = select_mock_group_by_target_depth(target_depth,
                                                          depth_df,
                                                          genome_mtx,
                                                          group_col=group_col)
            choice_sample, sample_mtx = select_random_sample_mtx(group_mtx)
            #Edge case where chosen group has average depth > target but chosen
            # sample depth < target -> change rarefaction depth to match 
            # sample depth. 
            rarefaction_depth = min(depth_df.loc[choice_sample,'Depth'],target_depth)
        elif choice_method == 'replicate':
            replicates_mtx = select_mock_replicates_by_target_depth(target_depth,
                                                                    depth_df,
                                                                    genome_mtx,
                                                                    replicate_col=replicate_col)
            choice_sample, sample_mtx = select_random_sample_mtx(replicates_mtx)
            rarefaction_depth = target_depth
        else:
            raise ValueError("Unrecognized option for choice_method; must be 'group' or 'replicate'.")
        #Make unique sample identifiers (in case choice sample is used for more than
        # one target depth)
        rarefaction_sample_identifier = "{0}_s{1}".format(choice_sample,i+1)
        mock_mtx_to_rarefy.loc[:,rarefaction_sample_identifier] = sample_mtx
        rarefaction_depths[i] = rarefaction_depth
    return mock_mtx_to_rarefy, rarefaction_depths

###============================================================================###
# Quasi-simulation: re-noising helper functions  
###============================================================================###

def generate_per_gene_noise(dispersion,gene_counts,
                            n_iterations=1,
                            random_state=None):
    """
    Given an array of gene counts, generate Gaussian noise per-gene with 
        mean of 0 (symmetrical) and variance parameterized using each gene's 
        expression level.
    
    :param dispersion: float, required. Dispersion estimate to use for 
        generating per-gene variance for Gaussian noise. Variance for a gene 
        with count C is given by the negative binomial mean-variance law 
        with var = C + dispersion * C^2
    :param gene_counts: array-like of int, required. Gene counts to re-noise. 
    :param n_iterations: int, optional. default 1. Number of noise iterations 
        to generate. The final noise array of shape (n_iteraitons, n_genes) 
        will be summed into per_gene_noise of shape (1,n_genes).  
    :param random_state: {None, int, numpy.random.Generator}. If provided, 
        seed noise generation for reproducibility.  

    :return noise_vector: array-like of shape (1,n_genes) containing per-gene 
        noise. 
    """
    from scipy.stats import norm
    size = (n_iterations,len(gene_counts))
    #For a given sample: means of noise are 0 (symmetric noise), variances are 
    # generated using mean-var trend of representative MAG;
    # i.e. var = mean + disp*mean^2. note scipy.stats.norm is parameterized
    # using std, so we take the square root here. 
    gene_wise_sigma = np.sqrt(gene_counts + dispersion*(gene_counts**2))
    #Use random_state for reproducibility if provided
    if random_state:
        noise_vector = norm(loc=0,scale=gene_wise_sigma).rvs(size=size,random_state=random_state)
    else:
        noise_vector = norm(loc=0,scale=gene_wise_sigma).rvs(size=size)
    #Collapse into 1D array if n_iterations > 1
    # if n_iterations > 1:
    noise_vector = noise_vector.sum(axis=0) 
    return noise_vector

def renoise_gene_counts(dispersion,gene_counts,
                            n_iterations=1,
                            random_state=None):
    """
    Given an array of gene counts, generate per_gene noise and return 
        renoised counts.  
    
    :param dispersion: float, required. Dispersion estimate to use for 
        generating per-gene variance for Gaussian noise. Variance for a gene 
        with count C is given by the negative binomial mean-variance law 
        with var = C + dispersion * C^2
    :param gene_counts: array-like of int, required. Gene counts to re-noise. 
    :param n_iterations: int, optional. default 1. Number of noise iterations 
        to generate. The final noise array of shape (n_iteraitons, n_genes) 
        will be summed into per_gene_noise of shape (1,n_genes).  
    :param random_state: {None, int, numpy.random.Generator}. If provided, 
        seed noise generation for reproducibility. Note that using an int 
        instead of a numpy rng object is not recommended and resulting noise 
        vectors might be artificially similar across samples 

    :return noised_counts: array-like of shape (1,n_genes) containing renoised
        counts. After incorporating noise, counts are converted back to positive
        integers (i.e. genes with negative noise of greater magnitude than original
        count -> 0). 
    """
    noise_vector = generate_per_gene_noise(dispersion,gene_counts,
                                           n_iterations=n_iterations,
                                           random_state=random_state)
    n_genes = len(gene_counts)
    noised_counts = gene_counts+noise_vector
    noised_counts = np.rint(np.maximum(np.zeros(n_genes),noised_counts)).astype(int)
    return noised_counts
    
def renoise_counts_table(dispersion,counts_df,
                            n_iterations=1,
                            random_state=None):
    """
    Given an array of gene counts, generate per_gene noise and return 
        renoised counts.  
    
    :param dispersion: float, required. Dispersion estimate to use for 
        generating per-gene variance for Gaussian noise. Variance for a gene 
        with count C is given by the negative binomial mean-variance law 
        with var = C + dispersion * C^2
    :param gene_counts: array-like of int, required. Gene counts to re-noise. 
    :param n_iterations: int or array-like, optional. default 1. Number of noise 
        iterations to generate. 
    :param random_state: {None, int, numpy.random.Generator}. If provided, 
        seed noise generation for reproducibility. Note that using an int 
        instead of a numpy rng object is not recommended and resulting noise 
        vectors might be artificially similar across samples 

    :return noised_counts: array-like of shape (1,n_genes) containing renoised
        counts. After incorporating noise, counts are converted back to positive
        integers (i.e. genes with negative noise of greater magnitude than original
        count -> 0). 
    """
    noised_counts_df = pd.DataFrame(index=counts_df.index,
                            columns=counts_df.columns)
    for i,sample_id in enumerate(counts_df.columns):
        gene_counts = counts_df[sample_id]
        if isinstance(n_iterations,int):
            noised_counts = renoise_gene_counts(dispersion,gene_counts,
                                                n_iterations=n_iterations,
                                                random_state=random_state)
        else:
            try: 
                sample_n_iterations = n_iterations[i]
            except: 
                raise ValueError("n_iterations must be an int or an array-like " \
                                 "of same length as number of samples in counts_df.")
            noised_counts = renoise_gene_counts(dispersion,gene_counts,
                                                n_iterations=sample_n_iterations,
                                                random_state=random_state)
        noised_counts_df[sample_id] = noised_counts
    return noised_counts_df

def compare_original_to_noised_counts_df(original_counts,
                                         noised_counts,
                                         n_iterations):
    """
    Comparison metrics between original and noised counts tables that 
    contain corresponding samples
    
    :param original_counts: Genes x samples, required. 
    :param noised_counts: Genes x samples, required. 
    :param n_iterations: int or array-like, required. 
    """
    #Check sample correspondence
    if not original_counts.columns.isin(noised_counts.columns).all():
        raise ValueError("Samples (columns) do not correspond for provided counts tables.")
    original_vs_noised = pd.DataFrame(index=original_counts.columns,
                                 columns=['original_depth','noised_depth',
                                          'original_detection','noised_detection',
                                          'n_iterations'])
    #Depth metrics - sum of counts 
    original_vs_noised['original_depth'] = original_counts.sum(axis=0)
    original_vs_noised['noised_depth'] = noised_counts.sum(axis=0)
    #Detection metrics - fraction of nonzero counts 
    original_vs_noised['original_detection'] = (original_counts>0).mean(axis=0)
    original_vs_noised['noised_detection'] = (noised_counts>0).mean(axis=0)
    original_vs_noised['n_iterations'] = n_iterations
    return original_vs_noised


###============================================================================###
# Visualization helper functions  
###============================================================================###

def plot_unit_line(ax,axlim=(0,10)):
    ax.plot(axlim,axlim,color='k',linestyle='dashed',zorder=0)
    return ax

def mean_variance_scatterplot(genome_mtx,
                            depth_normalize=True,
                            log_transform=True,
                            show_scatter=True,
                            scatter_alpha=0.4,
                            line_plot='disp_quadratic',
                            line_alpha=1,
                            ci=95,ci_alpha=0.4,
                            bootstrap_axis=1,
                            ax=None,figsize=(2,4),
                            xlim=(-0.5,6),
                            ylim=(-0.5,10.5),
                            tick_increment=2,
                            plot_unit=False,
                            line_color=MTX_colors.gradient_five_blue[3],
                            point_color=MTX_colors.gradient_five_blue[2],
                            zorder=0,random_seed=42):
    #Make new fig/ax if ax not provided:
    if not ax: 
        fig, ax = plt.subplots(1,1,figsize=figsize)

    #Get gene-wise mean and variance along with MME dispersion estimates:
    by_gene_mean_var = genome_counts_mean_variance_df(genome_mtx,
                                                      depth_normalize=depth_normalize,
                                                      log_transform=log_transform,
                                                      estimate_dispersions=True)
    #Set mean_col and var_col based on log_transform
    if log_transform:
        mean_col = 'log_mean'
        var_col = 'log_var'
    else:
        mean_col = 'mean'
        var_col = 'var'
    #Create scatter plot:
    if show_scatter:
        ax = sns.scatterplot(by_gene_mean_var,
                            x=mean_col,y=var_col,ax=ax,
                            color=point_color,alpha=scatter_alpha,linewidth=0,zorder=zorder,
                            s=5
                            )
    #Line plot options:

    #Using median genewise dispersion estimate to plot 
    # negative binomial variance 
    if line_plot == 'disp_quadratic':
        # max_obs = genome_mtx.max().max()
        max_mean_obs = by_gene_mean_var['mean'].max()
        #Generate range of x values for disp quadratic line 
        med_dispersion = by_gene_mean_var['dispersion'].median()
        #Start xrange close to 0 in linear space so trend line does 
        # will start at 1 
        xrange = np.logspace(-2,np.log10(max_mean_obs),num=100) 
        var_pred = xrange + med_dispersion*xrange**2 #NB variance using dispersion median
        if log_transform:
            ax.plot(np.log10(xrange+1),np.log10(var_pred+1),
                    color=line_color,
                    linewidth=1,
                    alpha=line_alpha,
                    zorder=zorder) #log scale var_pred
        else:
            ax.plot(xrange,var_pred,
                    color=line_color,
                    linewidth=1,
                    alpha=line_alpha,
                    zorder=zorder) #linear scale var_pred
        #Optionally CI interval using bootstraps 
        if ci:
            #Use bootstraps to get CI bounds for dispersion estimates 
            disp_bootstrap_df, disp_ci_upper, disp_ci_lower = estimate_dispersion_ci(genome_mtx,
                                                                                     depth_normalize=depth_normalize,
                                                                                     ci=ci,random_seed=random_seed,
                                                                                     bootstrap_axis=bootstrap_axis)
            ci_upper = xrange + disp_ci_upper*xrange**2
            ci_lower = xrange + disp_ci_lower*xrange**2
            if log_transform:
                ax.fill_between(x=np.log10(xrange+1),
                                y1=np.log10(ci_upper+1),
                                y2=np.log10(ci_lower+1),
                                color=line_color,
                                linewidth=0,
                                alpha=ci_alpha,
                                zorder=zorder)
            else:
                ax.fill_between(x=xrange,
                                y1=ci_upper,
                                y2=ci_lower,
                                color=line_color,
                                linewidth=0, 
                                alpha=ci_alpha,
                                zorder=zorder)

    #Options for just showing ordinary least squares mean dispersion fits; 
    # just uses seaborn regplot 
    elif line_plot == 'ols':
        #Just use seaborn regplot to generate line/CI
        #Set ci parameter based on show_ci option
        if ci==0:
            ci = None
        ax = sns.regplot(by_gene_mean_var,
                        x=mean_col,y=var_col,
                        ax=ax,
                        color=line_color,
                        scatter=False,
                        ci=ci,
                        line_kws={'alpha':1,'zorder':zorder,'linewidth':1.5})
    elif line_plot == 'none':
        pass
    else: 
        raise ValueError("Unrecognized value for line_plot; must be one of {'disp_quadratic','ols','none'}.")
    if plot_unit:
        plot_unit_line(ax)
    #Standardize axes limits 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #Standardize ticks using tick_increment
    # lim[1]//2*2+2 sets the highest tick to the greatest multiple of 2
    # that is less than lim[1]; +2 sets the range upper limit to include 
    # this multiple
    ax.set_xticks(range(0,int(xlim[1]//2*2+2),2))
    ax.set_yticks(range(0,int(ylim[1]//2*2+2),2))
    return ax
    