#! /usr/bin/env python

"""
synth_data_utils.py
Utility functions for basic loading, subsetting, and visualization of synth_mgx_mtx datasets.
Evan Lee
Last update: 09/12/23 
"""

###====================================================================================###
### Imports  
###====================================================================================###
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os, re, sys 
import importlib 

import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from MTX_utils import benchmarking, MTX_colors

#Global variables for default color palettes and lists of datasets names
MTX_bar_palette = MTX_colors.MTX_bar_palette
MTX_point_palette = MTX_colors.MTX_point_palette
zhang_6datasets = ['null-bug','true-exp-low','true-exp-med','true-exp','true-combo-bug-exp','true-combo-dep-exp']
zhang_5TE_datasets = ['true-exp-low','true-exp-med','true-exp','true-combo-bug-exp','true-combo-dep-exp']
###====================================================================================###
### Data loading and manipulation   
###====================================================================================###

def load_single_dataset(dataset,datatype,
                        synth_dir="synth/synth_mgx_mtx",
                        synth_original_dir = "synth/synth_mgx_mtx_original",
                        depth_normalize=False):
    """
    Load a single dataset datatype from standardized directory structure.  
    @param dataset: {'null-bug','true-exp-low','true-exp-med','true-exp',
                        'true-combo-bug-exp','true-combo-dep-exp'}
    @param datatype: {'mgx_abunds','mtx_abunds','bug_abunds','mgx_bug_abunds','metadata','mtx_spiked'}, required. 
    The feature data or metadata to load.
    """
    if datatype == 'mtx_spiked': #mtx_spiked are in synth_mgx_mtx_original 
        fpath = os.path.join(synth_original_dir,"{0}.mtx_spiked.tsv".format(dataset))
        spiked_features = pd.read_csv(fpath,sep='\t',names=["feature","direction"]).set_index('feature')
        return spiked_features
    else: 
        fpath = os.path.join(synth_dir,"{0}.{1}.tsv".format(dataset,datatype))
        data = pd.read_csv(fpath,sep='\t',index_col=0)
        if depth_normalize and datatype in ['mgx_abunds','mtx_abunds','bug+abunds','mgx_bud_abunds']: #scale each count so total scaled counts have equal sum/ depth across all samples 
            depths = data.sum(axis=0)
            data = data*depths.max()/depths
        return data 


def load_datasets(dataset_name,depth_normalize=False,
                    synth_dir="synth/synth_mgx_mtx",
                    synth_original_dir = "synth/synth_mgx_mtx_original",
                    spiked_datasets_to_load=['mtx']):
    """Load MTX synth datasets from pre-existing directories and file names. 
    TODO: optional loading of bug spiked features 

    @param dataset_name: str for which files of the form {dataset_name}.{mgx/mtx/bug}_abunds.tsv exist in synth_dir.
    @param synth_dir: path to directory containing mgx_abunds, mtx_abunds, bug_abunds, and metadata .tsv files 
    @praram synth_original_dir: path to directory containing spiked features lists (.mtx_spiked.tsv)

    @return datasets: dictionary, mapping keys ['mgx','mtx','bug','metadata','mtx_spiked'] to corresponding pandas DataFrames representing input data.
    """

    
    #Read mgx/mtx/bug counts and metadata from synth_mgx_mtx 
    mgx_fpath = os.path.join(synth_dir,"{0}.mgx_abunds.tsv".format(dataset_name))
    mtx_fpath = os.path.join(synth_dir,"{0}.mtx_abunds.tsv".format(dataset_name))
    metadata_fpath = os.path.join(synth_dir,"{0}.metadata.tsv".format(dataset_name))
    bug_fpath = os.path.join(synth_dir,"{0}.bug_abunds.tsv".format(dataset_name))

    mgx_df = pd.read_csv(mgx_fpath,sep='\t',index_col=0)
    mtx_df = pd.read_csv(mtx_fpath,sep='\t',index_col=0)
    metadata_df = pd.read_csv(metadata_fpath,sep='\t',index_col=0)
    bug_df = pd.read_csv(bug_fpath,sep='\t',index_col=0)
    
    #Read in spiked features from synth_mgx_mtx_original directory 
    spiked_datasets = {}
    for spiked_dataset in spiked_datasets_to_load:
        spiked_dataset_handle = "{0}_spiked".format(spiked_dataset)
        spiked_fpath = os.path.join(synth_original_dir,"{0}.{1}_spiked.tsv".format(dataset_name,spiked_dataset))
        spiked_df = pd.read_csv(spiked_fpath,sep='\t',index_col=0,names=["direction"])
        spiked_datasets[spiked_dataset_handle] = spiked_df
    # if not (re.search('null',dataset)):
    # mtx_spiked_fpath = os.path.join(synth_original_dir,"{0}.mtx_spiked.tsv".format(dataset_name))
    # mtx_spiked = pd.read_csv(mtx_spiked_fpath,sep='\t',index_col=0,names=["direction"])
    
    if depth_normalize: #scale each count so total scaled counts have equal sum/ depth across all samples 
        mgx_df = mgx_df*metadata_df.loc["MGX_SeqDepth"].max()/metadata_df.loc["MGX_SeqDepth"]
        mtx_df = mtx_df*metadata_df.loc["MTX_SeqDepth"].max()/metadata_df.loc["MTX_SeqDepth"]
        bug_seq_depths = bug_df.sum(axis=0)
        bug_df = bug_df * bug_seq_depths.max()/bug_seq_depths
    
    datasets = {"mgx":mgx_df,"mtx":mtx_df,"bug":bug_df,"metadata":metadata_df}
    for spiked_dataset_handle in spiked_datasets:
        datasets[spiked_dataset_handle] = spiked_datasets[spiked_dataset_handle]
    return datasets 

def min_med_max_abundance_bugs(bug_df,method='mean'):
    #Extract bug names for representative lowest, medium, and highest abundance bugs in bug_df.
    n_bugs = len(bug_df)
    if method=='mean':
        min_bug = bug_df.mean(axis=1).sort_values().index[0]
        med_bug = bug_df.mean(axis=1).sort_values().index[n_bugs//2]
        max_bug = bug_df.mean(axis=1).sort_values().index[-1]
    elif method == 'median':
        min_bug = bug_df.median(axis=1).sort_values().index[0]
        med_bug = bug_df.median(axis=1).sort_values().index[n_bugs//2]
        max_bug = bug_df.median(axis=1).sort_values().index[-1]
    else: 
        raise ValueError("Unrecognized method. Please select from {'mean','median'}.")
    return [min_bug,med_bug,max_bug]

def bug_filter_data(bugs,datasets):
    """
    For datasets (mgx,mtx,spiked features, etc.), extract the subset of features corresponding to each bug in bugs. 

    @param: bugs: array-like, contains list of bugs which will have subsetted data extracted 
    @param: datasets: array-like of DataFrames with index of "BUGXXXX_GROUPYYYY"-style features 

    @return: bug_filtered_datasets: dictionary with keys of bugs and values of lists containing subsetted data for each dataset in datasets.
    """
    bug_filtered_datasets={}
    for bug in bugs: 
        filtered_datasets = []
        for dataset in datasets:
            filtered_data = dataset.loc[dataset.index.str.contains(bug)]
            filtered_datasets.append(filtered_data)
        bug_filtered_datasets[bug] = filtered_datasets
    return bug_filtered_datasets

def format_original_synth_datasets(mgx_original, bug_original, mtx_original):
    """
    Format synthetic datasets from Zhang et al. for compatibility with various methods.

    Takes input mgx and mtx abundance data of gene-level counts as genes x 
        samples, extracts metadata and returns counts only mgx/mtx data 
        and metadata as DataFrames. 
    @param mgx_original, mtx_original: pd.DataFrame, required. Gene-level 
        counts data and 'Phenotype' and 'SeqDepth' metadata, genes x samples. 
    @param bug_original: pd.DataFrame, optional. Genome-level counts data
        and 'Phenotype' and 'SeqDepth' metadata, rows x samples. 
    
    @return mgx_df, mtx_df: pd.DataFrame. Gene-level counts data without 
        metadata features.
    @return bug_df: pd.DataFrame
    @return metadata_df: pd.DataFrame, contains features 'Phenotype', 
        'MGX_SeqDepth', and 'MTX_SeqDepth'. 
    """
    metadata_df = pd.DataFrame(index=["Phenotype","MGX_SeqDepth","MTX_SeqDepth"],
                           columns=mtx_original.columns)
    metadata_df.loc[["Phenotype"],:] = mgx_original.loc[["Phenotype"]]
    metadata_df.loc[["MGX_SeqDepth"],:] = mgx_original.loc[["SeqDepth"],:].values
    metadata_df.loc[["MTX_SeqDepth"],:] = mtx_original.loc[["SeqDepth"],:].values
    #Remove metadata features from counts tables
    mgx_df = mgx_original.drop(index=["Phenotype","SeqDepth"])
    bug_df = bug_original.drop(index=["Phenotype","SeqDepth"])
    mtx_df = mtx_original.drop(index=["Phenotype","SeqDepth"])
    #Remove index/column names from original Zhang data (each index is named #)
    mgx_df = mgx_df.rename_axis(None, axis=0)
    mtx_df = mtx_df.rename_axis(None, axis=0)
    bug_df = bug_df.rename_axis(None, axis=0)
    mgx_df = mgx_df.rename_axis(None, axis=1)
    mtx_df = mtx_df.rename_axis(None, axis=1)
    bug_df = bug_df.rename_axis(None, axis=1)

    return mgx_df,bug_df, mtx_df, metadata_df

def generate_mgx_bug_abunds(mgx_df,bug_df):
    """Generate a gene-level DataFrame (mgx_bug_abunds) where values are genome-level
        abundances. Needed for MTXmodel M5 as dna_input_data. 
    """
    mgx_bug_df = pd.DataFrame(index=mgx_df.index,columns=mgx_df.columns)
    for bug in bug_df.index:
        bug_data = bug_df.loc[bug,:] 
        mgx_bug_features = mgx_df.index[mgx_df.index.str.contains(bug)]
        mgx_bug_fill = pd.concat([bug_data]*len(mgx_bug_features),axis=1).T
        mgx_bug_fill.index = mgx_bug_features
        mgx_bug_df.loc[mgx_bug_features,:] = mgx_bug_fill
    return mgx_bug_df


###====================================================================================###
### Data Filtering   
###====================================================================================###

def abundance_partition_bug_df(bug_df,method='median',n_quantiles=10):
    """
    Partition bug_df into non-overlapping quantiles of bugs ranked by their relative abundance. 
    
    @param bug_df: pd.DataFrame, bugs x samples. Values should correspond to bug aggregated counts or absolute/relative abundances. 
                    counts/ abundance valiues will be sample sum scaled (column normalized) (i.e. correcting for sequencing depth or total bacterial load).
    @param method: {'median','mean','nonzero_median','nonzero_mean','prevalence'}, default='median'. 
                    Method for aggregating bug relative abundances before ranking/partitioning into quantiles. 
                    'nonzero_median' and 'nonzero_mean' ignore zero values (and attempt to account for differences in prevalence between bugs). 
                    'prevalence' ranks bugs by the number of samples with counts > 0. 
    @param n_qunatiles: int, default=10. Number of quantiles to partition the ranked bugs.

    @return partitions: dict, keys are ints corresponding to quantiles [0,n_quantiles-1] and values are pd.Index objects corresponding to 
                        non-overlapping subsets of the bugs in the index of bug_df. 
    """
    bug_seq_depth = bug_df.sum()
    #Calculate effective relative abundances - bug contribution out of total library reads 
    depth_normalized_bug_df = bug_df/bug_seq_depth 
    #Rank bugs by effective relative abundance 
    if method == 'median':
        bugs_ranked = depth_normalized_bug_df.median(axis=1).sort_values()
    elif method == 'mean':
        bugs_ranked = depth_normalized_bug_df.mean(axis=1).sort_values()
    elif method == 'nonzero_median':
        bugs_ranked = depth_normalized_bug_df.replace(0,np.nan).median(axis=1,skipna=True).sort_values()
    elif method == 'nonzero_mean':
        bugs_ranked = depth_normalized_bug_df.replace(0,np.nan).mean(axis=1,skipna=True).sort_values()
    elif method == 'prevalence':
        prevalence_data = (depth_normalized_bug_df > 0)
        bugs_ranked = prevalence_data.sum(axis=1).sort_values()
    else:
        raise ValueError("Invalid method selection; please choose from {'median','mean','nonzero_median','nonzero_mean','prevalence'}.")
    #Partition into quantiles 
    n_bugs = len(bugs_ranked)
    partitions = {} #dict from quantile number (k) to quantile partition bugs as pd.Index (v) 
    for i in range(n_quantiles):
        partition_start, partition_end = int(np.floor(n_bugs*i/n_quantiles)),int(np.floor(n_bugs*(i+1)/n_quantiles))
        partition_bugs = bugs_ranked.index[partition_start:partition_end]
        partitions[i+1] = partition_bugs #1-indexed partition labels 
    return partitions

def bug_abundance_partition_datasets(bug_df,datasets,partition_rank_method='median',n_quantiles=10,
                                        match_type='feature_name',feature_lookup_table=None,
                                        show_quantile_feature_counts=False):
    """
    Partition datasets into non-overlapping subsets of data by their underlying bug abundance quantiles in bug_df.  

    @param bug_df: pd.DataFrame, bugs x samples. Values should correspond to bug aggregated counts or absolute/relative abundances. 
                    counts/ abundance valiues will be sample sum scaled (column normalized) (i.e. correcting for sequencing depth or total bacterial load).
    @param datasets: array-like of pd.DataFrame objects. For each dataset, partitions will be returned in the corresponding partitioned_datasets 
    @param partition_rank_method: {'median','mean','nonzero_median','nonzero_mean','prevalence'}, default='median'. 
                    Method for aggregating bug relative abundances before ranking/partitioning into quantiles. 
                    'nonzero_median' and 'nonzero_mean' ignore zero values (and attempt to account for differences in prevalence between bugs). 
                    'prevalence' ranks bugs by the number of samples with counts > 0. 
    @param n_qunatiles: int, default=10. Number of quantiles to partition the ranked bugs.

    @return partitions: dict, keys are ints corresponding to quantiles [0,n_quantiles-1] and values are pd.Index objects corresponding to 
                        non-overlapping subsets of the bugs in the index of bug_df. 
    """
    bug_abundance_partitons = abundance_partition_bug_df(bug_df,method=partition_rank_method,n_quantiles=n_quantiles)
    partitioned_datasets = {}
    # total_partition_dataset_sizes = [0]*len(bug_abundance_partitons)
    partition_dataset_sizes = np.zeros(shape=(len(bug_abundance_partitons),len(datasets)))
    for partition in bug_abundance_partitons: #Note - partitions are 1-indexed (for human readability later on)
        partition_datasets = []
        partition_bugs = bug_abundance_partitons[partition] 
        for j,dataset in enumerate(datasets):
            if match_type == 'feature_name':
                partition_bugs_re = "|".join(partition_bugs)
                partition_bugs_data = dataset.loc[dataset.index.str.contains(partition_bugs_re),:]
                partition_dataset_sizes[partition-1,j] += len(partition_bugs_data)
                partition_datasets.append(partition_bugs_data)
            elif match_type == 'lookup':
                if not feature_lookup_table:
                    raise ValueError("Please provide feature_lookup_table, which should contain features as the index and contain a column 'Bug' mapping them to a specific organism in bug_df.")
                else:
                    raise ValueError("Sorry I haven't implemented this yet oops.")
            else: 
                raise ValueError("'feature' is the only supported option for match_type currently.")
        partitioned_datasets[partition] = partition_datasets
    total_partition_dataset_sizes = partition_dataset_sizes.sum(axis=0)
    if show_quantile_feature_counts: #Print out number of features in each partition for each dataset 
        print("Number of features in each quantile (row) for each dataset (column):")
        print(partition_dataset_sizes)
    #Sanity check that partitioned dataset sizes are same size as original datasets 
    for i in range(len(datasets)):#Check that total number of features in each partition equals dataset size 
        assert(total_partition_dataset_sizes[i] == len(datasets[i]))

    return partitioned_datasets



###====================================================================================###
### Data Visualization   
###====================================================================================###

def violin_swarmplot(data,x,y,hue,split=True,dodge=False,figsize=(8,4),
                    violin_palette=MTX_bar_palette,swarm_palette=MTX_point_palette,ax=None,
                    show_legend=True):
    #Basic seaborn style overlaid violin and swarm plots. 
    if not ax:
        fig,ax = plt.subplots(1,1,figsize=figsize)
    ax = sns.violinplot(data,x=x,y=y,hue=hue,
               palette=violin_palette,zorder=0,split=split,ax=ax,inner=None)
    ax = sns.swarmplot(data,x=x,y=y,hue=hue,
              palette=swarm_palette,zorder=1,dodge=dodge,ax=ax,size=3.5)
    if show_legend:
        sns.move_legend(ax,"upper left",bbox_to_anchor=(1,1))
    else:
        ax.get_legend().remove()
    return ax

def bar_swarmplot(data,x,y,hue,dodge=False,figsize=(8,4),order=None,hue_order=None,
                    bar_palette=MTX_bar_palette,swarm_palette=MTX_point_palette,ax=None,
                    show_legend=True,color_error_bars=False):
    if not ax: 
        fig, ax = plt.subplots(figsize=figsize)
    #barplot
    ax = sns.barplot(data,x=x,y=y,hue=hue,order=order,hue_order=hue_order,
                    palette=bar_palette,zorder=0,dodge=dodge,ax=ax,
                    capsize=0.1,errorbar="sd",err_kws={'linewidth':1})
    #overlaid swarmplot 
    ax = sns.swarmplot(data,x=x,y=y,hue=hue,order=order,hue_order=hue_order,
              palette=swarm_palette,zorder=1,dodge=dodge,ax=ax,size=3.5)
    if show_legend:
        sns.move_legend(ax,"upper left",bbox_to_anchor=(1,1))
    else:
        ax.get_legend().remove()

    if color_error_bars:
        if hue_order:
            ordered_point_colors = [swarm_palette[group] for group in hue_order]
        else:
            ordered_point_colors = list(swarm_palette.values())
        #https://stackoverflow.com/questions/66064409/errorbar-colors-in-seaborn-barplot-with-hue
        patches = ax.patches
        lines_per_err = 2 #2 seems to work in seaborn 0.13.0 
        for i, line in enumerate(ax.get_lines()):
            line_idx = i%(lines_per_err*len(ordered_point_colors))
            newcolor = ordered_point_colors[line_idx//lines_per_err]
            line.set_color(newcolor)
    #Set y minimum to 0 
    ymin,ymax  = ax.get_ylim()
    ax.set_ylim(0,ymax)
    return ax
