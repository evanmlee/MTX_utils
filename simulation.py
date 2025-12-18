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
                                   estimate_dispersions=False):
    #If depth_normalize, scale all samples so that they have equal depths 
    # to highest depth sample
    if depth_normalize:
        scaling_factors = genome_mtx.sum().max()/genome_mtx.sum()
        genome_mtx = np.rint(genome_mtx*scaling_factors).astype(int)
    #Generate summary stats DF - indexed on genes, metrics as columns
    gene_summary_metrics = pd.DataFrame(index=genome_mtx.index)
    gene_summary_metrics['mean'] = genome_mtx.mean(axis=1)
    #Variance metrics calculation:
    gene_summary_metrics['var'] = genome_mtx.var(axis=1)
    gene_summary_metrics['std'] = genome_mtx.std(axis=1)
    #Log transform metrics into separate columns if applicable 
    if log_transform:
        for col in gene_summary_metrics.columns:
            log_col = 'log_{0}'.format(col)
            gene_summary_metrics[log_col] = np.log10(gene_summary_metrics[col]+log_pseudocount)
    #Use NB mean-variance relationship to estimate dispersions for each gene 
    # We use the method of moments method here even though it is invalid for 
    # underdispersed counts; we'll be representing dispersion across the 
    # genome with its median dispersion (c/w Gierlinkski et al Bioinformatics 2015)
    if estimate_dispersions:
        gene_summary_metrics['dispersion'] = (gene_summary_metrics['var']-\
                                              gene_summary_metrics['mean'])/\
                                                gene_summary_metrics['mean']**2
    return gene_summary_metrics

#Bootstrapping mtx profiles 
def bootstrap_mtx_sample(genome_mtx):
    n_samples = len(genome_mtx.columns)
    return genome_mtx.sample(n=n_samples,replace=False)

def estimate_dispersion_ci(genome_mtx,
                           depth_normalize=True,
                           n_bootstraps=1000,
                           ci=95, random_seed=0):
    if random_seed:
        np.random.seed(random_seed)
    #Standardize library sizes and reintegerize with rint
    if depth_normalize:
        scaling_factors = genome_mtx.sum().max()/genome_mtx.sum()
        genome_mtx = np.rint(genome_mtx*scaling_factors).astype(int)
    bootstrap_df = pd.DataFrame(columns=['dispersion_med','dispersion_mean'])
    #Bootstrap sample from genome_mtx, then calculate mean/var/estimate MME dispersion
    # Then take central tendency with median/mean
    
    for i in range(n_bootstraps):
        bootstrap_sample_mtx = bootstrap_mtx_sample(genome_mtx)
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
# Visualization helper functions  
###============================================================================###

def plot_unit_line(ax,axlim=(0,10)):
    ax.plot(axlim,axlim,color='k',linestyle='dashed',zorder=0)
    return ax

def mean_variance_scatterplot(genome_mtx,
                            depth_normalize=True,
                            log_transform=True,
                            show_scatter=True,
                            line_plot='disp_quadratic',
                            ci=95,ci_alpha=0.4,
                            ax=None,figsize=(4,4),
                            plot_unit=True,
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
                            color=point_color,alpha=0.4,linewidth=0,zorder=zorder,
                            )
    #Line plot options:

    #Using median genewise dispersion estimate to plot 
    # negative binomial variance 
    if line_plot == 'disp_quadratic':
        max_obs = genome_mtx.max().max()
        #Generate range of x values for disp quadratic line 
        med_dispersion = by_gene_mean_var['dispersion'].median()
        xrange = np.logspace(0,np.log10(max_obs+1),num=500)
        var_pred = xrange + med_dispersion*xrange**2 #NB variance using dispersion median
        if log_transform:
            ax.plot(np.log10(xrange+1),np.log10(var_pred+1),
                    color=line_color,
                    linewidth=1.5,
                    zorder=zorder) #log scale var_pred
        else:
            ax.plot(xrange,var_pred,
                    color=line_color,
                    linewidth=1.5,
                    zorder=zorder) #linear scale var_pred
        #Optionally CI interval using bootstraps 
        if ci:
            #Use bootstraps to get CI bounds for dispersion estimates 
            disp_bootstrap_df, disp_ci_upper, disp_ci_lower = estimate_dispersion_ci(genome_mtx,
                                                                                     depth_normalize=depth_normalize,
                                                                                     ci=ci,random_seed=random_seed)
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
    return ax
    