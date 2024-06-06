###====================================================================================###
# Imports 
###====================================================================================###

import pandas as pd
import numpy as np
import os
import re 
import warnings
import sys 
import warnings

import matplotlib.pyplot as plt 
import seaborn as sns 
from MTX_utils import bacteria_info, MTX_colors

###====================================================================================###
### Kallisto Output Processing 
###====================================================================================###
### Prepare kallisto counts x samples table:
def process_kallisto_output(barcode_sample_df,kallisto_output_dir,
        kallisto_merged_table_fpath='',counts_type='est_counts',force_int=False):
    '''Parse expected directory structure in kallisto_output_dir and generate a DataFrame mapping samples to kallisto counts. 

    @param barcode_sample_df: pd.DataFrame, required. Must have columns Barcode and SampleName. The entries in Barcode will be 
    used to map subdirectories of kallisto_output_dir containing individual sample abundance estimates to sample names in the 
    returned DataFrame. 
    @param kallisto_output_dir: path to directory containing kallisto outputs, required. Should contain subdirectories 
    named by barcode entries in barcode_sample_df. 
    @param kallisto_merged_table_fpath: path, optional. If provided, will write kallisto_merged_counts_df to the provided path. 
    @param counts_type: {'est_counts','tpm'}. Default 'est_counts'. Determines the feature to extract from the kallisto output
    abundance.tsv files. 
    @return kallisto_merged_counts_df: pd.DataFrame, locus tags x samples 
    '''
    #Check barcode_sample_df
    if barcode_sample_df.index.name != 'Barcode' and 'Barcode' not in barcode_sample_df.columns:
        raise ValueError('Provided barcode_sample_df missing column Barcode.')
    if 'SampleName' not in barcode_sample_df.columns:
        raise ValueError('Provided barcode_sample_df missing column SampleName.')
    #Find all barcode directories (each for one sample) in kallisto_output_dir
    barcode_subdirs = os.listdir(kallisto_output_dir)
    #Ignore hidden directories for list of sample directories
    barcode_subdirs = [subdir for subdir in barcode_subdirs if '.' not in subdir]
    #For each barcode sample directory, read abundance information and store in kallisto_merged_counts_df 
    for i,barcode in enumerate(barcode_subdirs): 
        barcode_subdir_path = os.path.join(kallisto_output_dir,barcode)
        subdir_abundance_path = os.path.join(barcode_subdir_path,'abundance.tsv')
        barcode_abundance = pd.read_csv(subdir_abundance_path,sep='\t')
        #initialize kallisto_mereged_counts_df 
        if i == 0:
            kallisto_merged_counts_df = pd.DataFrame(index=barcode_abundance.index,columns=barcode_subdirs)
            kallisto_feature_names = barcode_abundance['target_id']
        #Populate appropriate counts or tpm DataFrames based on counts_type 
        if counts_type == 'est_counts':
            kallisto_merged_counts_df[barcode] = barcode_abundance['est_counts']
        elif counts_type == 'tpm':
            kallisto_merged_counts_df[barcode] = barcode_abundance['tpm']
        else: 
            raise ValueError("Invalid option provided for counts_type, must be 'est_counts' or 'tpm'.")

    #Formatting of returned DataFrame: 
    #Row names -> feature names from 'target_id' column 
    kallisto_merged_counts_df.index = kallisto_feature_names
    #Substitute barcode column names with Sample Names from barcode_sample_df 
    if barcode_sample_df.index.name != 'Barcode':
        barcode_sample_df = barcode_sample_df.set_index('Barcode')
    #Map barcodes to sample names and rename columns in kallisto_merged_counts_df
    kallisto_sample_names = barcode_sample_df.loc[kallisto_merged_counts_df.columns,'SampleName']
    kallisto_merged_counts_df.rename(columns=dict(zip(kallisto_merged_counts_df.columns,kallisto_sample_names)),inplace=True)
    #Sort sample names by their appearance in barcode_sample_df 
    sorted_sample_names = barcode_sample_df['SampleName'][barcode_sample_df['SampleName'].isin(kallisto_merged_counts_df.columns)]
    kallisto_merged_counts_df = kallisto_merged_counts_df[sorted_sample_names]
    #Optional handling of float data from kallisto with force_int; np.ceil used to differentiate low confidence prob estimates (or low tpm)
    #from 0s. 
    if force_int:
        if counts_type == 'tpm':
            warnings.warn('Specified counts_type=tpm and force_int=True. Rounding TPMs to int, be aware this is not often desirable.')
        kallisto_merged_counts_df = np.ceil(kallisto_merged_counts_df).astype(int)

    #Write results if kallisto_merged_table_fpath is provided.  
    if kallisto_merged_table_fpath:
        #Determine if .csv, .tsv, or .txt fpath provided 
        if re.search(r'\.csv',kallisto_merged_table_fpath):
            kallisto_merged_counts_df.to_csv(kallisto_merged_table_fpath)
        elif re.search(r'\.tsv',kallisto_merged_table_fpath):
            kallisto_merged_counts_df.to_csv(kallisto_merged_table_fpath,sep='\t')
        elif re.search(r'\.txt',kallisto_merged_table_fpath):
            kallisto_merged_counts_df.to_csv(kallisto_merged_table_fpath,sep=' ')
        else: 
            raise ValueError('kallisto_merged_table_fpath should be a .csv, .tsv, or .txt file path.')
    return kallisto_merged_counts_df

###====================================================================================###
# Processed Output Loading 
###====================================================================================###
def load_single_kallisto_dataset(dataset,datatype,
                        base_dir='MG02_MGX_MTX',
                        depth_normalize=False):
    '''
    Load a single dataset datatype from standardized directory structure.  
    @param dataset: {'MG02_AC','MG02_AD'}
    @param datatype: {'mgx_abunds','mtx_abunds','bug_abunds','mgx_bug_abunds','metadata'}, required. 
    Spiked datasets removed as option for MG02/kallisto datasets.  
    The feature data or metadata to load.
    '''
    fpath = os.path.join(base_dir,'{0}.{1}.tsv'.format(dataset,datatype))
    data = pd.read_csv(fpath,sep='\t',index_col=0)
    if depth_normalize and datatype in ['mgx_abunds','mtx_abunds','bug_abunds','mgx_bud_abunds']: #scale each count so total scaled counts have equal sum/ depth across all samples 
        depths = data.sum(axis=0)
        data = data*depths.max()/depths
    return data 


def load_kallisto_datasets(dataset_name,depth_normalize=False,
                    MG02_dir='MG02_MGX_MTX'):
    '''Load kallisto datasets from pre-existing directories and file names. 

    @param dataset_name: str for which files of the form {dataset_name}.{mgx/mtx/bug}_abunds.tsv exist in MG02_dir.
    @param MG02_dir: path to directory containing mgx_abunds, mtx_abunds, bug_abunds, and metadata .tsv files 
    @praram synth_original_dir: path to directory containing spiked features lists (.mtx_spiked.tsv)

    @return datasets: dictionary, mapping keys ['mgx','mtx','bug','metadata','mtx_spiked'] to corresponding pandas DataFrames representing input data.
    '''

    
    #Read mgx/mtx/bug counts and metadata from synth_mgx_mtx 
    mgx_fpath = os.path.join(MG02_dir,'{0}.mgx_abunds.tsv'.format(dataset_name))
    mtx_fpath = os.path.join(MG02_dir,'{0}.mtx_abunds.tsv'.format(dataset_name))
    metadata_fpath = os.path.join(MG02_dir,'{0}.metadata.tsv'.format(dataset_name))
    bug_fpath = os.path.join(MG02_dir,'{0}.bug_abunds.tsv'.format(dataset_name))

    mgx_df = pd.read_csv(mgx_fpath,sep='\t',index_col=0)
    mtx_df = pd.read_csv(mtx_fpath,sep='\t',index_col=0)
    metadata_df = pd.read_csv(metadata_fpath,sep='\t',index_col=0)
    bug_df = pd.read_csv(bug_fpath,sep='\t',index_col=0)
    
    if depth_normalize: #scale each count so total scaled counts have equal sum/ depth across all samples 
        mgx_df = mgx_df*metadata_df.loc['MGX_SeqDepth'].max()/metadata_df.loc['MGX_SeqDepth']
        mtx_df = mtx_df*metadata_df.loc['MTX_SeqDepth'].max()/metadata_df.loc['MTX_SeqDepth']
        bug_seq_depths = bug_df.sum(axis=0)
        bug_df = bug_df * bug_seq_depths.max()/bug_seq_depths
    
    datasets = {'mgx':mgx_df,'mtx':mtx_df,'bug':bug_df,'metadata':metadata_df}
    return datasets 

def load_coproseq_results(coproseq_fpath='norm_percent.profile',bacteria_aliases={},
                        bacteria_info_df=[],info_names_col='index',
                            info_rename_col='index',drop_spike_ins=[]):
    '''Load coproseq results data into shared data format (taxa on rows, samples on columns). 

    @param coproseq_fpath: path to load coproseq results from. Required. Important: the index of the resulting 
    coproseq_df is set to the first column. 
    
    @param bacteria_aliases: dict, optional. If provided, replace organism identifiers in coproseq data as 
    read from file. Happens before any replacements using bacteria_info_df, so is useful for forcing 
    compatibility with entries in bacteria_info_df. 
    @param bacteria_info_df: pd.DataFrame. Optional. 
    Default behavior (no DataFrame provided): Returns table from coproseq_fpath as is, with index provided by 
    first column. 
    If provided, is used to map the taxon specifiers in the coproseq_df index to alternate values. 
    Must contain the columns specified by info_names_col and info_rename_col if they are provided. 
    When bacteria_info_df is provided, the returned coproseq_df is reindexed replacing values in 
    the original index (found in info_names_col) with corresponding values from info_rename_col.
    @param info_names_col: Label in bacteria_info_df, optional, default='index'. 
    Default behavior ('index'): Expects the values in the index of coproseq_df to be found in the index of 
    bacteria_info_df. 
    If provided: Matches values in the index of coproseq_df against the specified column in bacteria_info_df.  
    @param info_rename_col: Label in bacteria_info_df, optional, default='index'. 
    Default behavior ('index'): coproseq_df index values are replaced with corresponding entries in the 
    index of bacteria_info_df. 
    If provided: coproseq_df index values are replaced with corresponding entries in the specified column in bacteria_info_df. 

    @param drop_spike_ins=[]: array-like, optional. If provided, rows corresponding to entries in drop_spike_ins
    will be removed from the returned coproseq_df and relative abundances will be renormalized to the subset 
    of remaining organisms (such that the total of relative abundances for the subset is 1.) 

    @return coproseq_df: pd.DataFrame of relative abundances of each organism from CoProSeq data with
    organism identifiers as the index and sample names as the columns  
    '''
    #Read in coproseq_df from file
    coproseq_df = pd.read_csv(coproseq_fpath,sep='\t',index_col=0)
    #If bacteria_aliases is provided, replace names in coproseq_df index before any further processing with bacteria_info. 
    if len(bacteria_aliases) > 0:
        coproseq_df = coproseq_df.rename(index=bacteria_aliases)
    #If bacteria_info_df is provided, substitute coproseq organism identifiers based on info_names_col and info_rename_col 
    if len(bacteria_info_df) > 0: 
        if type(bacteria_info_df) != pd.DataFrame:
            raise ValueError('bacteria_info_df must be a pd.DataFrame')
        bacteria_info_coproseq_names_indexed = bacteria_info_df.reset_index(names='index').set_index(info_names_col)
        coproseq_names_in_bacteria_info = coproseq_df.index[coproseq_df.index.isin(bacteria_info_coproseq_names_indexed.index)]
        if len(coproseq_names_in_bacteria_info) == 0:
            raise ValueError('Current specification (info_names_col={0}) does not contain any of the index entries for coproseq_fpath.'.format(info_names_col))
        else: 
            coproseq_rename_values = bacteria_info_coproseq_names_indexed.loc[coproseq_names_in_bacteria_info,info_rename_col]
            coproseq_reindexed_df =  coproseq_df.reset_index(names='index')
            coproseq_reindexed_df['index'] = coproseq_reindexed_df['index'].replace(coproseq_rename_values)
            coproseq_reindexed_df = coproseq_reindexed_df.set_index('index')
            coproseq_df = coproseq_reindexed_df
    #Renormalize reported RAs without including spike-in data 
    if len(drop_spike_ins) > 0: 
        coproseq_df = coproseq_df.drop(index=drop_spike_ins)
        coproseq_df = coproseq_df/coproseq_df.sum(axis=0) #Recalculate RAs normalizing to sum of non-spike in species
    return coproseq_df

###====================================================================================###
### Data Summarization  
###====================================================================================###
def taxon_DE_fractions(results_df,bug_df,alpha=0.05,sig_label='qval',subset_to_tested=True,
                        all_features_df=[],logFC_label='coef'):
    '''Return basic statistics about the fraction of differentially expressed genes ('DE_Fraction')
    and fraction of DE genes that are upregulated (logFC > 0; 'Sig_Up_Fraction') for a results_df 
    containing differential expression results, with aggregation into organisms based on row labels
    in bug_df. 


    '''
    bug_DE_fractions_df = pd.DataFrame(index=bug_df.index,columns=['DE_fraction','n_DE','n_tested','n_features','n_sig_up','sig_up_fraction'])
    for bug in bug_df.index:
        bug_results_features = results_df.loc[results_df.index.str.contains(bug)]
        n_bug_tested_features = len(bug_results_features)
        bug_results_sig = bug_results_features[bug_results_features[sig_label]<=alpha]
        n_bug_sig_features = len(bug_results_sig)
        if subset_to_tested:
            DE_fraction = n_bug_sig_features/n_bug_tested_features
            bug_DE_fractions_df.loc[bug,'n_features'] = n_bug_tested_features
        else: 
            if len(all_features_df) == 0:
                raise ValueError('subset_to_tested=True but no DataFrame provided as all_features_df.')
            else: 
                n_bug_features = len(all_features_df.loc[all_features_df.index.str.contains(bug)])
                DE_fraction = n_bug_sig_features/n_bug_features
                bug_DE_fractions_df.loc[bug,'n_features'] = n_bug_features
        bug_DE_fractions_df.loc[bug,'n_DE'] = n_bug_sig_features
        bug_DE_fractions_df.loc[bug,'n_tested'] = n_bug_tested_features
        bug_DE_fractions_df.loc[bug,'DE_fraction'] = DE_fraction
        if n_bug_sig_features > 0:
            n_sig_up = len(bug_results_sig[bug_results_sig[logFC_label]>0])
            sig_up_fraction = len(bug_results_sig[bug_results_sig[logFC_label]>0])/n_bug_sig_features
        else:
            n_sig_up = 0
            sig_up_fraction = np.nan
        bug_DE_fractions_df.loc[bug,'n_sig_up'] = n_sig_up
        bug_DE_fractions_df.loc[bug,'sig_up_fraction'] = sig_up_fraction
    return bug_DE_fractions_df

###====================================================================================###
### Data Visualization  
###====================================================================================###

def volcano_plot(results_df,logFC_label='coef',pval_label='qval',
                hue_label='significant',
                alpha=0.05,
                ax=None,figsize=(4,4),title='',
                xlim=(-10,10),ylim=(-0.5,7),
                palette={True:sns.color_palette('colorblind')[0],False:MTX_colors.NS_gray},
                legend=True):
    '''Generate a volcano plot of differential testing results, plotting logFC vs -log(p_value) for
    each gene in results_df. 

    @param results_df: pd.DataFrame, required. Must contain columns specified by logFC_label and p_val_label. 
    @param logFC_label, pval_label: labels in results_df, required. Defaults 'coef' and 'qval' for compatibility
    with DE test results returned by Zhang MTXmodel R package. 

    '''
    if not ax: 
        fig, ax = plt.subplots(1,1,figsize=figsize)
    results_df = results_df.copy() #Make copy so manipulations/ extra variables are not added to original DataFrame
    implemented_hue_label_methods = ['significant','direction_significant']
    if hue_label not in results_df and hue_label not in implemented_hue_label_methods:
        raise ValueError('Specified hue_label {0} is not in results_df.'.format(hue_label))
    elif hue_label == 'significant':
        results_df['significant'] = results_df[pval_label] <=alpha
    elif hue_label == 'direction_significant':
        results_df['direction_significant'] = (results_df[pval_label] <=alpha).astype(int)
        down_results = results_df.loc[(results_df[logFC_label] < 0)].index
        results_df.loc[down_results,'direction_significant'] = results_df.loc[down_results,'direction_significant']*-1
        display(results_df['direction_significant'].unique())
        # significant_down_results = results_df.loc[(results_df['direction_significant'] == 1) & \
                                                    # (results_df[logFC_label] < 0)].index


    # display(results_df[hue_label])
    results_df['log_pval'] = -np.log10(results_df[pval_label])
    #Volcano plot scatter
    ax = sns.scatterplot(results_df.sort_values(hue_label),x=logFC_label,y='log_pval',hue=hue_label,
        #hue_order=(results_df[hue_label].unique().sort()),
        palette=palette,ax=ax)
    ax.set_xlabel('logFC')
    ax.set_ylabel('-log10 p-value')
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #Move legend to upper right corner outside of figure or remove if legend=False
    if legend:
        sns.move_legend(ax,'upper left',bbox_to_anchor=(1,1))
    else:
        ax.get_legend().remove()

def violin_stripplot(all_results_df,strip_results_df,logFC_label='coef',xlabel='strain',
                        pval_label='qval',hue_label='significant',
                        alpha=0.05,ax=None,figsize=(2,4),title='',violin_norm='count',
                        ylim=(-10,10),markersize=5,
                        palette={True:sns.color_palette('colorblind')[0],False:MTX_colors.NS_gray},
                        legend=True):
    if not ax: 
        fig, ax = plt.subplots(1,1,figsize=figsize)
    all_results_df = all_results_df.copy()
    strip_results_df = strip_results_df.copy()
    if hue_label not in all_results_df and hue_label != 'significant':
        raise ValueError('Specified hue_label {0} is not in results_df.'.format(hue_label))
    elif hue_label == 'significant':
        all_results_df['significant'] = all_results_df[pval_label] <=alpha
        strip_results_df['significant'] = strip_results_df[pval_label] <=alpha
    #Violin plot: all FC values in all_results_df 
    ax = sns.violinplot(all_results_df,x=xlabel,y=logFC_label,ax=ax,color='#DDDDDD',zorder=1,density_norm=violin_norm,inner=None,
                  linewidth=1,cut=0)
    if len(all_results_df[hue_label].unique())<=2 and hue_label=='significant': #Single position strip plot 
        ax = sns.swarmplot(strip_results_df,x=xlabel,y=logFC_label,hue=hue_label,
                            ax=ax,palette=palette,dodge=False,zorder=2,size=markersize)
    else:
        #If there are multiple levels in hue_label:
        #1.split results into significant and non-significant
        #2. Generate strip plots 
        swarm_results_sig = strip_results_df[strip_results_df[pval_label]<=alpha].sort_values(hue_label)
        swarm_results_ns = strip_results_df[strip_results_df[pval_label]>=alpha].sort_values(hue_label)
        hue_levels = swarm_results_ns[hue_label].unique()
        #ns_palette encodes different hue labels to support usage of dodge in sns.stripplot (i.e. different
        #x positions for each hue label) but applies the same NS_gray hue value for each label
        ns_palette = dict(zip(hue_levels,[MTX_colors.NS_gray]*len(hue_levels)))
        ax = sns.stripplot(swarm_results_ns,x=xlabel,y=logFC_label,ax=ax,hue=hue_label,hue_order=hue_levels,
                        palette=ns_palette,dodge=True,size=markersize,zorder=2)
        ax = sns.stripplot(swarm_results_sig,x=xlabel,y=logFC_label,ax=ax,hue=hue_label,hue_order=hue_levels,
                        palette=palette,dodge=True,size=markersize,zorder=3)
    #Standardize axes and title
    ax.set_xlabel('')
    ax.set_ylabel('logFC')
    ax.set_title(title)
    ax.set_ylim(ylim)
    #Move or remove legend 
    if legend:
        sns.move_legend(ax,'upper left',bbox_to_anchor=(1,1))
    else:
        if ax.get_legend():
            ax.get_legend().remove()
    return ax

def mcSEED_violin_stripplot(all_results_df,bacteria_info,mcSEED,strains=[],phenotype_regexps=[],
                            logFC_label='coef',pval_label='qval',alpha=0.1,violinplot_geneset='all',ax=None,
                            ylim=(-10,10),markersize=5,figsize=(2,4)):
    '''Generate a violin stripplot of differential expression test results showing specific mcSEED annotated
    genes from specified strains and pathways. 

    @param all_results_df: pd.DataFrame, required. DataFrame of differential expression test results. Must contain a 
    log-fold change analog (logFC_label) and a (FDR-adjusted) p-value column (pval_label) which will be used 
    respectively as the y-axis and hue-encoding of points in the resulting plot. 
    @param bacteria_info: pd.DataFrame, required. Used to map locus identifier prefixes in all_results_df and mcSEED 
    to the strain identifiers supplied in strains. 
    @param mcSEED: pd.DataFrame, required. Indexed on gene locus identifiers and must contain column 'Phenotype'
    which will be used to identify specific genes to plot in the stripplot 
    @param strains: array-like, optional, default []. List of identifiers from the 'organism' column in bacteria_info. 
    Must contain at least one entry or a ValueError will be raised. 
    @param phenotype_regexps: array-like, optional, default []. List of regular expressions which will be used to 
    match entries in the 'Phenotype' column of mcSEED. Each regular expression's set of matched genes will be 
    plotted as a separate hue (split into significant/non-significant test results) in the resulting plot. 
    @param logFC_label, pval_label: labels in results_df. Will be used as y-axis of violin/strip-plot and to hue genes
    by their significance 
    '''
    #Do not propagate changes to original results DataFrame 
    all_results_df = all_results_df.copy() 
    #Helper variables for supplied lists of strains and phenotypes
    n_strains = len(strains)
    n_phenotypes = len(phenotype_regexps)
    bacteria_plot_order_dict = dict(zip(strains,range(len(strains)))) #For maintaining order of violin plots 

    #QC supplied values
    if n_strains == 0:
        raise ValueError('Please supply a list of at least one strain identifiers in strains; currently empty.')

    #Set up mcSEED palette based on number of phenotypes provided 
    if n_phenotypes <= 6: #Use ordered paired hues from colorblind_subset
        mcSEED_palette = dict(zip(range(1,7),MTX_colors.colorblind_subset))
    elif n_phenotypes <= 12: #Use paired seaborn palette
        mcSEED_palette = dict(zip(range(1,11),MTX_colors.twelve_paired))
    else:     
        warnings.warn('Number of specified phenotypes exceeds palette length, colors will be repeated.')
        mcSEED_palette = dict(zip(range(1,n_phenotypes+1),sns.color_palette('colorblind',n_phenotypes)))
    mcSEED_palette[0] = MTX_colors.NS_gray #Add in no-significance color
    
    #Add strain and phenotype metadata to results_df 
    all_results_df['Strain'] = all_results_df.index.str.extract(r'(\w+)_\d+',expand=False).map(bacteria_info['organism'])
    #Use provided list of strains to extract locus tags from results_df corresponding to genes from those organisms
    #Locus prefixes for provided list of strains
    bacteria_lt_prefixes = [bacteria_info[bacteria_info['organism']==strain].index[0] for strain in strains]
    #Extract corresponding genes from all_results_df 
    bacteria_lt_re_pat = '|'.join(bacteria_lt_prefixes)
    bacteria_results_df = all_results_df.loc[all_results_df.index.str.contains(bacteria_lt_re_pat)]
    #Sort by provided list of strains 
    bacteria_results_df = bacteria_results_df.sort_values('Strain',key=lambda x:x.map(bacteria_plot_order_dict))
    #Filter bacteria_results to bacteria results with mcSEED annotations 
    mcSEED_results_df = bacteria_results_df[bacteria_results_df.index.isin(mcSEED.index)]
    #Encode mcSEED phenotypes from provided list of regexps
    bacteria_results_df['Phenotype'] = 0
    mcSEED_results_df['Phenotype'] = 0
    #stripplot results will contain: 
    #1. only those genes from mcSEED results matching one of the provided phenotype regular expressions
    #2. duplicate entries for locus tags/ DE test results that match multiple of the regular expressions 
    mcSEED_results_for_stripplot = pd.DataFrame(columns=mcSEED_results_df.columns)
    for j,pht in enumerate(phenotype_regexps):
        all_pht_loci = mcSEED[mcSEED['Phenotype'].str.contains(pht)]
        pht_loci_in_mcSEED_results = mcSEED_results_df.loc[mcSEED_results_df.index.isin(all_pht_loci.index)]

        pht_loci_in_mcSEED_results['Phenotype'] = j + 1
        mcSEED_results_for_stripplot = pd.concat((mcSEED_results_for_stripplot,pht_loci_in_mcSEED_results))
    #Choose background distribution of genes for violinplot 
    if violinplot_geneset == 'all':
        violinplot_results = bacteria_results_df
    elif violinplot_geneset == 'mcSEED':
        violinplot_results = mcSEED_results_df
    else:
        warnings.warn('Unrecognized option for violinplot_geneset; using all genes.')
        violinplot_results = bacteria_results_df
    ax = violin_stripplot(violinplot_results,mcSEED_results_for_stripplot,
                                 logFC_label=logFC_label,
                                pval_label=pval_label,
                                xlabel='Strain',
                                 hue_label='Phenotype',
                                alpha=alpha,ax=ax,violin_norm='area',
                                 legend=False,palette=mcSEED_palette,ylim=ylim,
                                 markersize=markersize,figsize=figsize)
    return ax 

def bar_swarmplot(data,x,y,hue,dodge=False,figsize=(8,4),order=None,hue_order=None,
                    bar_palette=MTX_colors.MG02_bar_palette.values(),
                    swarm_palette=MTX_colors.MG02_point_palette.values(),ax=None,
                    show_legend=True):
    if not ax: 
        fig, ax = plt.subplots(figsize=figsize)
    #barplot
    ax = sns.barplot(data,x=x,y=y,hue=hue,order=order,hue_order=hue_order,
                    palette=bar_palette,zorder=0,dodge=dodge,ax=ax,
                    capsize=0.1,errorbar='sd',err_kws={'linewidth':1})
    #overlaid swarmplot 
    ax = sns.swarmplot(data,x=x,y=y,hue=hue,order=order,hue_order=hue_order,
              palette=swarm_palette,zorder=1,dodge=dodge,ax=ax,size=3.5)
    if show_legend:
        sns.move_legend(ax,'upper left',bbox_to_anchor=(1,1))
    else:
        ax.get_legend().remove()
    #Set y minimum to 0 
    ymin,ymax  = ax.get_ylim()
    ax.set_ylim(0,ymax)
    return ax
