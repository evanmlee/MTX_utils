#! /usr/bin/env python

"""
kallisto_data_utils.py
Utility functions for processing kallisto/bowtie2 quantification outputs into
counts tables (genes x samples). Also contains some generic data visualization
functions (bar_swarmplot, volcano_plot, violin_stripplot) for plotting 
quantitative data and DE results, respectively. 
Evan Lee
Last update: 04/03/25
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
### Kallisto Output Processing 
###============================================================================###
### Prepare kallisto counts x samples table:
def process_kallisto_output(barcode_sample_df,kallisto_output_dir,
        kallisto_merged_table_fpath='',counts_type='est_counts',
        force_int=False,int_method='rint',
        barcode_label='Barcode',samplename_label='SampleName'):
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
    if barcode_sample_df.index.name != barcode_label and barcode_label not in barcode_sample_df.columns:
        raise ValueError('Provided barcode_sample_df missing column {0} specified by barcode_col.'.format(barcode_label))
    if barcode_sample_df.index.name != samplename_label and samplename_label not in barcode_sample_df.columns:
        raise ValueError('Provided barcode_sample_df missing column {0} specified by samplename_col.'.format(samplename_label))
    #Format handling of barcode_sample_df: reset index so barcode_label and samplename_label are columns, not index 
    if barcode_sample_df.index.name == barcode_label or barcode_sample_df.index.name == samplename_label:
        barcode_sample_df = barcode_sample_df.reset_index()
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
    if barcode_sample_df.index.name != barcode_label:
        barcode_sample_df = barcode_sample_df.set_index(barcode_label)
    #Map barcodes to sample names and rename columns in kallisto_merged_counts_df
    kallisto_sample_names = barcode_sample_df.loc[kallisto_merged_counts_df.columns,samplename_label]
    kallisto_merged_counts_df.rename(columns=dict(zip(kallisto_merged_counts_df.columns,kallisto_sample_names)),inplace=True)
    #Sort sample names by their appearance in barcode_sample_df 
    sorted_sample_names = barcode_sample_df[samplename_label][barcode_sample_df[samplename_label].isin(kallisto_merged_counts_df.columns)]
    kallisto_merged_counts_df = kallisto_merged_counts_df[sorted_sample_names]
    #Optional handling of float data from kallisto with force_int; np.ceil used to differentiate low confidence prob estimates (or low tpm)
    #from 0s. 
    if force_int:
        #Warn if forcing int for tpms
        if counts_type == 'tpm':
            warnings.warn('Specified counts_type=tpm and force_int=True. Rounding TPMs to int, be aware this is not often desirable.')
        #Default behavior: round to nearest integer to prevent low kallisto 
        #pseudocounts (potential multi/mismapping) from rounding up. 
        # Also supports ceil/floor if specified by int_method:
        if int_method == 'ceil':
            kallisto_merged_counts_df = np.ceil(kallisto_merged_counts_df).astype(int)
        elif int_method == 'floor':
            kallisto_merged_counts_df = np.floor(kallisto_merged_counts_df).astype(int)
        else:
            kallisto_merged_counts_df = np.rint(kallisto_merged_counts_df).astype(int)

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

###============================================================================###
### Bowtie2/Samtools Counts Processing 
###============================================================================###
def process_bowtie2_output(barcode_sample_df,bowtie2_output_dir,
        counts_file_name_suffix='_microbial.txt',
        bowtie2_merged_table_fpath='',counts_type='est_counts',
        barcode_label='Barcode',samplename_label='SampleName'):
    '''Process bowtie2 and samtools idxcounts output into a counts DataFrame indexed on locus tags and with samples 
    as columns. 

    @param barcode_sample_df: pd.DataFrame, required. Must have columns Barcode and SampleName. The entries in Barcode will be 
    used to map subdirectories of kallisto_output_dir containing individual sample abundance estimates to sample names in the 
    returned DataFrame. 
    @param bowtie2_output_dir: path to directory containing kallisto outputs, required. Must contain individual sample 
    index counts outputs of the format {bowtie2_output_dir}/{barcode}{counts_file_name_suffix}
    '''
    #Check barcode_sample_df for corresponding columns 
    if barcode_sample_df.index.name != barcode_label and barcode_label not in barcode_sample_df.columns:
        raise ValueError('Provided barcode_sample_df missing column {0} specified by barcode_col.'.format(barcode_label))
    if barcode_sample_df.index.name != samplename_label and samplename_label not in barcode_sample_df.columns:
        raise ValueError('Provided barcode_sample_df missing column {0} specified by samplename_col.'.format(samplename_label))
    #Format handling of barcode_sample_df: reset index and set index to barcode_label
    if barcode_sample_df.index.name == samplename_label:
        barcode_sample_df = barcode_sample_df.reset_index()
    if barcode_sample_df.index.name != barcode_label:
        barcode_sample_df = barcode_sample_df.set_index(barcode_label)
    #Find all barcode directories (each for one sample) in kallisto_output_dir
    all_barcode_files = [fname for fname in os.listdir(bowtie2_output_dir) \
                        if re.search(counts_file_name_suffix,fname)]
    #Determine delimiter to use for counts file reading 
    if re.search(r'\.csv',counts_file_name_suffix):
        delim = ','
    elif re.search(r'\.txt|\.tsv',counts_file_name_suffix):
        delim = '\t'
    else:
        raise ValueError('unknown file suffix specified by counts_file_name_suffix. Must contain .csv, .tsv, or .txt.')
    bowtie2_merged_counts_df = pd.DataFrame()
    for i,barcode_fname in enumerate(all_barcode_files):
        barcode = re.match('([ACGT]+[-_][ACGT]+){0}'.format(counts_file_name_suffix),barcode_fname).group(1)
        bowtie2_sample_counts = pd.read_csv(os.path.join(bowtie2_output_dir,barcode_fname),
                sep=delim,names=['target_id','length','counts','unmapped_counts'])
        #Convert to format compatible with pd.concat across all samples: indexed on target_id and 
        #columns named by barcode
        bowtie2_sample_counts = bowtie2_sample_counts.set_index('target_id')\
                                    .loc[:,['counts']].rename(columns={'counts':barcode})
        bowtie2_merged_counts_df = pd.concat((bowtie2_merged_counts_df,bowtie2_sample_counts),axis=1)
    #Rename columns from barcodes to sample identifiers
    bowtie2_sample_names = barcode_sample_df.loc[bowtie2_merged_counts_df.columns,samplename_label]
    bowtie2_merged_counts_df.rename(columns=dict(zip(bowtie2_merged_counts_df.columns,bowtie2_sample_names)),inplace=True)
    #Sort sample names by their appearance in barcode_sample_df 
    sorted_sample_names = barcode_sample_df[samplename_label][barcode_sample_df[samplename_label].isin(bowtie2_merged_counts_df.columns)]
    bowtie2_merged_counts_df = bowtie2_merged_counts_df[sorted_sample_names]
    
    #Write results if bowtie2_merged_table_fpath is provided
    if bowtie2_merged_table_fpath:
        if re.search(r'\.csv',bowtie2_merged_table_fpath):
            bowtie2_merged_counts_df.to_csv(bowtie2_merged_table_fpath)
        elif re.search(r'\.tsv',bowtie2_merged_table_fpath):
            bowtie2_merged_counts_df.to_csv(bowtie2_merged_table_fpath,sep='\t')
        elif re.search(r'\.txt',bowtie2_merged_table_fpath):
            bowtie2_merged_counts_df.to_csv(bowtie2_merged_table_fpath,sep=' ')
        else: 
            raise ValueError('kallisto_merged_table_fpath should be a .csv, .tsv, or .txt file path.')
    return bowtie2_merged_counts_df


###============================================================================###
# Processed Output Loading 
###============================================================================###
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

###============================================================================###
### Preparing DE input tables: benchmarking datasets (BG01) 
###============================================================================###

#Helper function for collapsing mgx abunds into bug df - OK 
def bug_abunds_from_mgx(mgx_df,locus_prefixes=[]):
    bug_df = pd.DataFrame(index=locus_prefixes,columns=mgx_df.columns)
    for locus_prefix in locus_prefixes:
        mixture_bug = mgx_df.loc[mgx_df.index.str.contains(locus_prefix)].sum()
        bug_df.loc[locus_prefix,:] = mixture_bug
    return bug_df

def select_mixture_data(mixture, all_MGX, all_MTX, all_metadata,
                        locus_prefixes=[],
                        metadata_as_cols=False):
    #Return mixture_MGX, mixture_MTX, mixture_bug, and mixture_metadata.
    #locus_prefixes must be provided in order to generate bug_abunds. 
    # locus_prefixes should contain substrings of gene identifiers encoding
    # their genome of origin (ex. BUG0001 for genes BUG0001_0001, BUG0001_0002)
    #mixture_metadata just contains Phenotype as a row which is binary encoding of Aran vs Glc.
    samples_md = all_metadata.loc[all_metadata['Mixture']==mixture]
    samples = samples_md.index
    #Subset counts data; intended format is genes x samples 
    mixture_mgx = all_MGX.loc[:,samples]
    mixture_mtx = all_MTX.loc[:,samples]
    #Collapse mgx into bug abunds 
    mixture_bug = bug_abunds_from_mgx(mixture_mgx,locus_prefixes=locus_prefixes)
    #Convert media to phenotype factor
    phenotypes = (samples_md.loc[:,['Media']]=='PCDM+Aran').astype(int).rename(columns={'Media':'Phenotype'})
    if metadata_as_cols:
        mixture_metadata = phenotypes.copy() #Do not transpose if overridden, results in samples x Phenotype 
    else: 
        mixture_metadata = phenotypes.T #Transpose by default - results in metadata features x samples 
    return mixture_mgx, mixture_bug, mixture_mtx, mixture_metadata

def select_phenotype_data(phenotype,mgx,bug,mtx,metadata,phenotype_label='Phenotype'):
    #Subset all four inputs to samples by phenotype in metadata
    if phenotype_label in metadata.columns:
        mgx = mgx.loc[:,metadata[phenotype_label]==phenotype] 
        bug = bug.loc[:,metadata[phenotype_label]==phenotype] 
        mtx = mtx.loc[:,metadata[phenotype_label]==phenotype] 
        metadata = metadata.loc[metadata[phenotype_label]==phenotype,:]
    elif phenotype_label in metadata.index:
        mgx = mgx.loc[:,metadata.loc[phenotype_label,:]==phenotype] 
        bug = bug.loc[:,metadata.loc[phenotype_label,:]==phenotype] 
        mtx = mtx.loc[:,metadata.loc[phenotype_label,:]==phenotype] 
        metadata = metadata.loc[:,metadata.loc[phenotype_label,:]==phenotype]
    else:
        raise ValueError('Metadata is missing provided phenotype_label')
    return mgx,bug,mtx,metadata

def zero_sample_data(zero_mgx,zero_bug,zero_mtx,zero_metadata,
                     zero_sample_size=0,phenotype=0): 
    """Randomly sample non-colonized data with replacement from provided mgx/bug/mtx/md. 
    
    @param zero_mgx,zero_bug,zero_mtx,zero_metadata: pd.DataFrame, required. Counts
        and metadata from which to sample with replacement. These data are assumed 
        to be non-colonized, though any counts and metadata can be provided to 
        sample from. 
    @param zero_sample_size: int, default 0. Sample size for non-colonized data. 
    @param phenotype: int, default 0. Fill value to assign phenotype with in 
        resulting zero_samples_md.
    @return zero_samples_mgx,zero_samples_bug,zero_samples_mtx: pd.DataFrame, 
        counts data for random sample of non-colonized data 
    @return zero_samples_md: pd.DataFrame, metadata for random sample. 
    
    """
    zero_samples = np.random.choice(zero_metadata.columns,size=zero_sample_size)
    zero_samples_mgx = zero_mgx.loc[:,zero_samples]
    zero_samples_bug = zero_bug.loc[:,zero_samples]
    zero_samples_mtx = zero_mtx.loc[:,zero_samples]
    zero_samples_md = zero_metadata.loc[:,zero_samples]
    #1. give zero samples unique identifiers (_P{phenotype}_Z{zero sample number}) - OK
    zero_samples_unique = [sample+"_P{0}_Z{1}".format(phenotype,i+1) \
                        for i,sample in enumerate(zero_samples_md.columns)]
    #2. Re-index counts and metadata with unique identifiers (by renaming samples) - OK
        #Dirctionary approach using rename does not handle duplicate labels (i.e. duplicates will get mapped to only one 'unique' label)
        #Explicitly setting index and column values from array to handle duplicate labels
    zero_samples_md.columns = zero_samples_unique
    zero_samples_mgx.columns = zero_samples_unique
    zero_samples_bug.columns = zero_samples_unique
    zero_samples_mtx.columns = zero_samples_unique
    #3. Harcode phenotype values in metadata for which group these zero samples are being added to -TODO
    zero_samples_md.loc['Phenotype',:] = phenotype
    return zero_samples_mgx,zero_samples_bug,zero_samples_mtx,zero_samples_md

def zero_inflate_mixture_data(all_MGX,all_MTX,all_metadata,
                              mixture,
                              zero_mixture_ID='Pco0%',
                              zero_sample_size=0,
                              locus_prefixes=[],
                              metadata_as_cols=False):
    """For a specified mixture, select relevant counts and metadata and spike-in
    non-colonized samples to emulate incomplete prevalence. 

    TODO: Add options for more control over zero sampling (filtering phenotype, etc)
    @param all_MGX, all_MTX, all_metadata: pd.DataFrame, required. Counts and metadata 
        with features as rows and samples as columns. 
    @param mixture: Mixture identifier in all_metadata, required; returned 
        datasets will contain colonized samples at this RA level zero inflated
        with non-colonized samples. 
    @param zero_mixture_ID: Mixture identifier in all_metadata, default 'Pco0%'. 
        The non-colonized samples to spike-in will be sampled with replacement 
        from this mixture level. 
    @param zero_sample_size: int, default 0. Number of non-colonized samples to 
        spike-in.
    @param metadata_as_cols: bool, default False. If True, metadata will be in
     format samples x features.  
    
    @return mixture_md_ZI: pd.DataFrame containing the zero-inflated metadata. 
        The non-colonized samples will have unique identifiers containing the 
        original zero-mixture sample ID with a '_Z{XX}' suffix. 
        The zero-inflated samples will have forced phenotype values in each 
        group (i.e. the non-colonized samples are sampled from both Arabinan and 
        Glucose 0% samples, but will have 'Phenotype' reassigned)
    @ return mixture_mgx_ZI, mixture_bug_ZI,mixture_mtx_ZI: counts DataFrames
        for the zero-inflated dataset
    """
    #Select mixture data to zero-inflate 
    mixture_mgx, \
    mixture_bug, \
    mixture_mtx, \
    mixture_metadata = select_mixture_data(mixture,
                                            all_MGX,all_MTX,
                                            all_metadata,
                                            locus_prefixes=locus_prefixes,
                                            metadata_as_cols=False)
    #Select zero_mixture_ID data to sample spike-ins from:
    zero_mgx, \
    zero_bug, \
    zero_mtx, \
    zero_metadata = select_mixture_data(zero_mixture_ID,
                                        all_MGX,all_MTX,
                                        all_metadata,
                                        locus_prefixes=locus_prefixes,
                                        metadata_as_cols=False)
    #Empty DataFrames to populate with mixture and ZI samples before returning
    mixture_md_ZI, \
    mixture_mgx_ZI, \
    mixture_bug_ZI, \
    mixture_mtx_ZI = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    #For each phenotype, select mixture data and generate/add accompanying zero data 
    for phenotype in (1,0):
        #Select mixture/phenotype data 
        mixture_mgx_pht, \
        mixture_bug_pht, \
        mixture_mtx_pht, \
        mixture_md_pht = select_phenotype_data(phenotype,
                                                mixture_mgx,
                                                mixture_bug,
                                                mixture_mtx,
                                                mixture_metadata)
        #Generate random sample (with replacement) from all 12 Pco0 samples for this phenotype group
        zero_samples_mgx,\
        zero_samples_bug,\
        zero_samples_mtx,\
        zero_samples_md = zero_sample_data(zero_mgx,zero_bug,zero_mtx,zero_metadata,
                                            zero_sample_size=zero_sample_size,
                                            phenotype=phenotype)
        #4. Combine mixture and zero sample data 
        mixture_md_ZI_pht = pd.concat((mixture_md_pht,zero_samples_md),axis=1)
        mixture_mgx_ZI_pht = pd.concat((mixture_mgx_pht,zero_samples_mgx),axis=1)
        mixture_bug_ZI_pht = pd.concat((mixture_bug_pht,zero_samples_bug),axis=1)
        mixture_mtx_ZI_pht = pd.concat((mixture_mtx_pht,zero_samples_mtx),axis=1)
        #5. Add into combined ZI dataset for both phenotpyes 
        mixture_md_ZI = pd.concat((mixture_md_ZI,mixture_md_ZI_pht),axis=1)
        mixture_mgx_ZI = pd.concat((mixture_mgx_ZI,mixture_mgx_ZI_pht),axis=1)
        mixture_bug_ZI = pd.concat((mixture_bug_ZI,mixture_bug_ZI_pht),axis=1)
        mixture_mtx_ZI = pd.concat((mixture_mtx_ZI,mixture_mtx_ZI_pht),axis=1)
    return mixture_mgx_ZI,mixture_bug_ZI,mixture_mtx_ZI,mixture_md_ZI

def convert_metadata_deseq_to_mpra(metadata_df):
    """Convert a metadata table from DESeq2-compatible format to MPRAnalyze-compatible format.
    """
    #Transpose to samples as rows
    metadata_df = metadata_df.T
    #Add SampleID covariate (to be used as lib.factor so that each sample is normalized separately)
    metadata_df['SampleID'] = metadata_df.index
    #Remove index name
    metadata_df.index.name = ''
    #Add Barcode covariate (currently unused)
    metadata_df['Barcode'] = 1
    return metadata_df

def write_tables(method_input_dir,dataset_handle,
                 mixture_mgx,mixture_bug,mixture_mtx,mixture_metadata,overwrite_files=False):
    """Write counts and metadata to defined directory structure for a given DE input dataset. 
    """
    mixture_mgx_fpath = os.path.join(method_input_dir,"{0}.mgx_abunds.tsv".format(dataset_handle))
    mixture_bug_fpath = os.path.join(method_input_dir,"{0}.bug_abunds.tsv".format(dataset_handle))
    mixture_mtx_fpath = os.path.join(method_input_dir,"{0}.mtx_abunds.tsv".format(dataset_handle))
    mixture_md_fpath = os.path.join(method_input_dir,"{0}.metadata.tsv".format(dataset_handle))

    for fpath, table in zip([mixture_mgx_fpath,mixture_bug_fpath,mixture_mtx_fpath,mixture_md_fpath],
                               [mixture_mgx,mixture_bug,mixture_mtx,mixture_metadata]):
        if not os.path.exists(fpath) or overwrite_files:
            table.to_csv(fpath,sep='\t')

def write_tables_MPRA_metadata(method_input_dir,dataset_handle,
                 mixture_mgx,mixture_bug,mixture_mtx,mixture_metadata,
                   overwrite_files=False,mgx_fname='mgx_abunds.tsv',
                   generate_bug_mtx=False):
    """Write counts and metadata to defined directory structure for a given DE input dataset,
        including MPRAnalyze formatted metadata (features as columns with a lib.factor compatible column) 
    """
    mixture_mgx_fpath = os.path.join(method_input_dir,"{0}.{1}".format(dataset_handle,mgx_fname))
    mixture_bug_fpath = os.path.join(method_input_dir,"{0}.bug_abunds.tsv".format(dataset_handle))
    mixture_mtx_fpath = os.path.join(method_input_dir,"{0}.mtx_abunds.tsv".format(dataset_handle))
    mixture_md_fpath = os.path.join(method_input_dir,"{0}.metadata.tsv".format(dataset_handle))
    mixture_mpra_md_fpath = os.path.join(method_input_dir,"{0}.MPRA_metadata.tsv".format(dataset_handle))
    #Generate mpra-formatted metadata 
    mixture_mpra_md = convert_metadata_deseq_to_mpra(mixture_metadata)
    for fpath, table in zip([mixture_mgx_fpath,mixture_bug_fpath,mixture_mtx_fpath,mixture_md_fpath,mixture_mpra_md_fpath],
                               [mixture_mgx,mixture_bug,mixture_mtx,mixture_metadata,mixture_mpra_md]):
        if not os.path.exists(fpath) or overwrite_files:
            table.to_csv(fpath,sep='\t')
    #Also write bug_mtx_abunds (one feature per bug, with values as sum of bug's mtx genes)
    if generate_bug_mtx:
        mixture_bug_mtx = mixture_mtx.copy()
        #Use generic organism_locus re pattern to extract organism identifiers
        mixture_bug_mtx['locus_base'] = mixture_bug_mtx.index.str.extract(r'(\w+)_\d+',expand=False)
        #Aggregate by organism
        mixture_bug_mtx = mixture_bug_mtx.groupby('locus_base').agg('sum')
        #File path check and write:
        mixture_bug_mtx_fpath = os.path.join(method_input_dir,"{0}.bug_abunds_mtx.tsv".format(dataset_handle))
        if not os.path.exists(mixture_bug_mtx_fpath) or overwrite_files:
            mixture_bug_mtx.to_csv(mixture_bug_mtx_fpath,sep='\t')

###============================================================================###
### Preparing DE input tables: in vitro crossfeeding datasets (BG04) 
###============================================================================###

#Helper function for getting counts for specific sample lists 
def get_samples_mgx_mtx(samples_list,mgx_df,mtx_df):
    #Check that samples are in counts tables; assumes samples on cols of counts tables 
    for sample in samples_list:
        if sample not in mgx_df.columns or sample not in mtx_df.columns:
            raise ValueError("Provided sample label {0} not in counts dataframes.".format(sample))
    return mgx_df.loc[:,samples_list],mtx_df.loc[:,samples_list]
#Helper function for getting sample identifiers matching metadata 
def get_mixture_media_time_samples(all_metadata,mixture,media,time):
    samples_list = all_metadata.loc[(all_metadata['Mixture']==mixture) & \
                                    (all_metadata['Media']==media) & \
                                    (all_metadata['Time']==time),:].index
    return samples_list
#Get mgx, mtx counts by mixture/media/time sample metadata
def get_mixture_media_time_counts(all_metadata,mgx_df,mtx_df,
                                 mixture,media,time):
    return get_samples_mgx_mtx(get_mixture_media_time_samples(all_metadata,mixture,media,time),
                              mgx_df,mtx_df)
#Generate a 'Phenotype' x samples DataFrame from two condition strings
# Treatment and reference are 1, 0 in 'Phenotype' and inferred automatically
# using the factor_encoding functions below.   
def generate_comparison_metadata_df(all_metadata,condition_str1,condition_str2):
    #Split condition strings into corresponding metadata 
    mix1, media1, time1 = condition_str1.split('-')
    mix2, media2, time2 = condition_str2.split('-')
    samples_condition1 = get_mixture_media_time_samples(all_metadata,mix1,media1,time1)
    samples_condition2 = get_mixture_media_time_samples(all_metadata,mix2,media2,time2)
    condition_encoding = condition_str_factor_encoding(condition_str1,condition_str2)
    comparison_metadata = pd.DataFrame()
    #Populate Phenotype metadata feature using condition_encoding
    comparison_metadata.loc['Phenotype',samples_condition1] = condition_encoding[condition_str1]
    comparison_metadata.loc['Phenotype',samples_condition2] = condition_encoding[condition_str2]
    comparison_metadata = comparison_metadata.astype(int)
    return comparison_metadata

#Encode timepoints as a {0,1} factor to determine DE treatment/reference
def time_factor_encoding(time1,time2):
    #Convert time strs ({XX}h) to int values 
    time1_value, time2_value = int(time1[:time1.index('h')]),int(time2[:time2.index('h')])
    #Use > comparison to determine phenotype values 
    #Encode higher time point as treatment (1)
    if time1_value > time2_value:
        time_encoding = {time1:1,time2:0}
    elif time2_value > time1_value:
        time_encoding = {time1:0,time2:1}
    #If timepoints are equal, give both same value of 1; when multiplying other factors in, 
    # one condition will be set to 0
    else: 
        time_encoding = {time2:1}
    return time_encoding
#Encode media as a {0,1} factor to determine DE treatment/reference
def media_factor_encoding(media1,media2):
    #media1 and media2: {'Arn','Aos','Glc'}
    # Intended reference levels: Arn > Aos > Glc
    #Aos vs Glc case (or non-Arn self vs self)
    if media1 != 'Arn' and media2 != 'Arn':
        media_encoding = {'Aos':1,'Glc':0}
    #Arn vs Glc case (or non-Aos self vs self)
    elif media1 != 'Aos' and media2 != 'Aos':
        media_encoding = {'Arn':1,'Glc':0}
    #Arn vs Aos case 
    elif media1 != 'Glc' and media2 != 'Glc':
        media_encoding = {'Arn':1,'Aos':0}
    return media_encoding
#Encode inoculum as a {0,1} factor to determine DE treatment/reference
def mixture_factor_encoding(mixture1,mixture2):
    #Mixtures: {'Pco', 'Mmu', 'Pco+Mmu'}
    #Cannot be Pco vs. Mmu; must be self vs self or Pco+Mmu vs single organism
    #Handle self vs self case: return 1 for both groups 
    if mixture1 == mixture2:
        mixture_encoding = {mixture1:1}
    elif mixture1 == 'Pco+Mmu':
        mixture_encoding = {mixture1:1,mixture2:0}
    elif mixture2 == 'Pco+Mmu':
        mixture_encoding = {mixture1:0,mixture2:1}
    else:
        raise ValueError("Cannot set factor levels for mixture1: \
                        {0} and mixture 2: {1}".format(mixture1,mixture2))
    return mixture_encoding
#Convert strings specifying sample metadata to {0,1} factor for DE testing 
def condition_str_factor_encoding(condition_str1,condition_str2):
    # Encode two mix-media-time strings to final {0,1} factors for use with DE testing
    # If both conditions strings are the same, raise ValueError 
    if condition_str1 == condition_str2:
        raise ValueError("Both provided condition strings are the same.")
    
    mix1, media1, time1 = condition_str1.split('-')
    mix2, media2, time2 = condition_str2.split('-')
    mixture_encoding = mixture_factor_encoding(mix1,mix2)
    media_encoding = media_factor_encoding(media1,media2)
    time_encoding = time_factor_encoding(time1,time2)
    #Multiply {0,1} factors to give end comparison reference levels:
    condition1_factor = mixture_encoding[mix1]*media_encoding[media1]*time_encoding[time1]
    condition2_factor = mixture_encoding[mix2]*media_encoding[media2]*time_encoding[time2]
    if condition1_factor == condition2_factor:
        warning_message = ("Both conditions contain one mixture/media/time metadata which is encoded as lower. "
                            "Prioritizing media > mixture > time differences for final factors.")
        warnings.warn(warning_message)
        #Test code - prioritizing certain metadata over others (media>mixture>time)
        # If media factors differ, final encoding uses media encodings
        if media_encoding[media1] != media_encoding[media2]:
            return {condition_str1:media_encoding[media1],
                   condition_str2:media_encoding[media2]}
        elif mixture_encoding[mix1] != mixture_encoding[mix2]:
            return {condition_str1:mixture_encoding[mix1],
                   condition_str2:mixture_encoding[mix2]}
        else: 
            return {condition_str1:time_encoding[time1],
                   condition_str2:time_encoding[time2]}
    else:
        return {condition_str1:condition1_factor,condition_str2:condition2_factor}

###============================================================================###
### Data Summarization  
###============================================================================###
def mgx_to_relative_abundance(mgx_df,bacteria_info,spike_ins=[],
                                locus_tag_prefix_re_pat=r'([\w]+)_\d+',
                                index_on_organism=False):
    '''Convert a table of metagenomic counts (by gene) into relative abundances by organism for each sample. 

    @param mgx_df: DataFrame, reqruied. Indexed on locus tags and samples as columns. 
    @param bacteria_info: DataFrame, required. Must be indexed on locus tag prefixes for genes in 
    mgx_df. Must contain column organism if spike_ins is provided. See bacteria_info.py.
    @param spike-in: array-like, optional. Locus tag prefixes corresponding to spike-in strains which
    should not be included in relative abundance calculation/ normalization. 
    @param locus_tag_prefix_re_pat: str, optional. Regular expression pattern to extract locus tag prefixes from 
    gene identifiers. Default assumes form of {locus tag prefix}_{integer gene index}. 
    @param index_on_organism: bool, default False. Index returned relative abundance DataFrame on 
    organism identifiers instead of locus tag prefixes 
    '''
    #Add temporary strain column 
    mgx_df['locus_tag_prefix'] = mgx_df.index.str.extract(locus_tag_prefix_re_pat,expand=False)
    if len(spike_ins) > 0:
        #Check format of spike_ins by checking if they match entries in locus_tag_prefix or bacteria_info['organism']
        lt_spike_ins = [lt for lt in spike_ins if lt in mgx_df['locus_tag_prefix'].unique()]
        organism_spike_ins = [lt for lt in spike_ins if lt in bacteria_info['organism'].unique()]    
        #Filter out lt_spike_ins
        mgx_df = mgx_df.loc[~(mgx_df['locus_tag_prefix'].isin(lt_spike_ins))]
        #Filter out organism_spike_ins - map locus_tag_prefixes using bacteria_info, filter out if in organism_spike_ins
        mgx_df = mgx_df.loc[~(mgx_df['locus_tag_prefix'].map(bacteria_info['organism']).isin(organism_spike_ins))]
    mgx_sample_cols = mgx_df.columns[~(mgx_df.columns.isin(['locus_tag_prefix']))]
    mgx_depths_per_sample = mgx_df[mgx_sample_cols].sum(axis=0)
    norm_mgx = mgx_df.loc[:,mgx_sample_cols]/mgx_depths_per_sample
    norm_mgx['locus_tag_prefix'] = norm_mgx.index.str.extract(locus_tag_prefix_re_pat,expand=False)
    organism_ra_df = norm_mgx.groupby('locus_tag_prefix').agg(sum)
    if index_on_organism:
        organism_ra_df.index = bacteria_info.loc[organism_ra_df.index,'organism']
    return organism_ra_df
    

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

###============================================================================###
### Data Visualization - generic visualization functions  
###============================================================================###

def volcano_plot(results_df,logFC_label='coef',pval_label='qval',
                hue_label='significant',alpha=0.05,
                replace_zero_pval=True,zero_pval_value=2.225074e-308,
                ax=None,figsize=(4,4),title='',
                xlim=(-10,10),ylim=(-0.5,7),markersize=5,linewidth=0,
                downsample=False,downsample_class=0,downsample_fraction=0.1,
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

    if replace_zero_pval:
        results_df[pval_label] = results_df[pval_label].replace(0,zero_pval_value)

    results_df['log_pval'] = -np.log10(results_df[pval_label])

    #Handle downsampling if specified - for points with 
    # where hue_label==downsample_class, use df.sample 
    # to downsample points to reduce redundant elements in plot 
    if downsample:
        data_to_downsample = results_df[(results_df[hue_label]==downsample_class) & 
                                        (np.abs(results_df[logFC_label])<0.2)] #downsample points with low logFC and matching downsample_class
        data_as_is = results_df[~results_df.index.isin(data_to_downsample.index)]
        np.random.seed(42)
        downsampled = data_to_downsample.sample(frac=downsample_fraction)
        results_df = pd.concat((data_as_is,downsampled))

    #Volcano plot scatter
    ax = sns.scatterplot(results_df.sort_values(hue_label),x=logFC_label,y='log_pval',hue=hue_label,
        palette=palette,ax=ax,s=markersize,linewidth=linewidth)

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
    return ax, results_df

def violin_stripplot(all_results_df,strip_results_df,logFC_label='coef',xlabel='strain',
                        pval_label='qval',hue_label='significant',
                        alpha=0.05,ax=None,figsize=(2,4),title='',violin_norm='count',
                        ylim=(-10,10),markersize=5,linewidth=0,edgecolor='k',
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
    ax = sns.violinplot(all_results_df,x=xlabel,y=logFC_label,ax=ax,color='#DDDDDD',zorder=1,
                        density_norm=violin_norm,inner=None,linecolor='#000000',
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
        hue_levels = strip_results_df[hue_label].unique()
        #ns_palette encodes different hue labels to support usage of dodge in sns.stripplot
        # (to align with significant strippplot, i.e. different x positions for 
        # each hue label) but applies the same NS_gray hue value for all

        #Try to infer NS gray hue to use from palette, or use default
        if 0 in palette:
            NS_gray = palette[0]
        else:
            NS_gray = MTX_colors.NS_gray
        ns_palette = dict(zip(hue_levels,[NS_gray]*len(hue_levels)))
        if len(swarm_results_ns) > 0:
            ax = sns.stripplot(swarm_results_ns,x=xlabel,y=logFC_label,ax=ax,hue=hue_label,hue_order=hue_levels,
                        palette=ns_palette,dodge=True,size=markersize,zorder=2,
                        linewidth=linewidth,edgecolor=edgecolor)
        if len(swarm_results_sig) > 0:
            ax = sns.stripplot(swarm_results_sig,x=xlabel,y=logFC_label,ax=ax,hue=hue_label,hue_order=hue_levels,
                        palette=palette,dodge=True,size=markersize,zorder=3,
                        linewidth=linewidth,edgecolor=edgecolor)
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


def bar_swarmplot(data,x,y,hue,dodge=False,figsize=(8,4),order=None,hue_order=None,
                  palette={},bar_palette={},swarm_palette={},
                  use_violin=False,
                    bar_alpha=1,swarm_alpha=1,markersize=5,ax=None,
                    show_legend=True,marker_col=None,marker_dict={}):
    if not ax: 
        fig, ax = plt.subplots(figsize=figsize)
    #Palette handling - allow either separate bar and swarm palettes or one shared palette. 
    # If no palettes passed, use default three colors from MG02 palettes.
    if len(palette) == 0 and (len(bar_palette)==0 and len(swarm_palette)==0):
        bar_palette = MTX_colors.MG02_bar_palette.values()
        swarm_palette = MTX_colors.MG02_point_palette.values()
    #If both bar and swarm palettes provided, use as is.
    elif len(bar_palette) > 0 and len(swarm_palette) > 0:
        pass #Use specified bar and swarm palettes as is
    #If only palette provided, use for both bar and swarm; 
    # then set bar_alpha=0.6 for visual distinction 
    elif len(palette) > 0:
        bar_palette = palette
        swarm_palette = palette
        bar_alpha = 0.6
    else:
        raise ValueError("Unknown palette options, please provide either palette or both bar_palette and swarm_palette.")

    #barplot
    if use_violin:
        ax = sns.violinplot(data,x=x,y=y,hue=hue,order=order,hue_order=hue_order,
                         palette=bar_palette,
                         zorder=0,dodge=False,ax=ax,cut=0,density_norm='count',
                         inner=None,alpha=bar_alpha,legend=True,)
    else:
        ax = sns.barplot(data,x=x,y=y,hue=hue,order=order,hue_order=hue_order,
                    palette=bar_palette,zorder=0,dodge=dodge,ax=ax,
                    capsize=0.1,errorbar='sd',
                    err_kws={'linewidth':0.5,'color':'#000000'},
                    alpha=bar_alpha,
                    legend=True)
    #overlaid swarmplot 
    if not marker_col:
        ax = sns.swarmplot(data,x=x,y=y,hue=hue,order=order,hue_order=hue_order,
              palette=swarm_palette,zorder=1,dodge=dodge,ax=ax,size=markersize,
              alpha=swarm_alpha)
    else: 
        #marker_dict maps values in marker_col to matplotlib marker str specifications 
        # markers: https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
        #For each marker_col value, make separate swarmplot

        #This has not been tested in cases where x/y/hue do not have all unique values
        # represented in each marker_data subset, which may cause axis misalignment 
        for metadata in marker_dict:
            marker = marker_dict[metadata]
            marker_data = data[data[marker_col]==metadata]
            ax = sns.swarmplot(marker_data,x=x,y=y,hue=hue,order=order,hue_order=hue_order,
              palette=swarm_palette,zorder=1,dodge=dodge,ax=ax,size=markersize,marker=marker,
              alpha=swarm_alpha)
    # if ax.get_legend():
    if show_legend:
        sns.move_legend(ax,'upper left',bbox_to_anchor=(1,1))
    else:
        ax.get_legend().remove()
    #Set y minimum to 0 - deprecated in case horizontal barplot
    # ymin,ymax  = ax.get_ylim()
    # ax.set_ylim(0,ymax)
    return ax

###=========================================================================###
### Data Visualization - application-specific visualization functions  
###=========================================================================###

def counts_regplot(counts_df,sample1,sample2,
                    locus_prefix='',xlim=(),ylim=(),
                    depth_normalize=False,ax=None,
                    figsize=(4,4),plot_unit=False,
                    anno_PCC=False,anno_slope=False):
    '''Generate a regression plot between gene-level counts for two different 
        samples. 
    @param counts_df: pd.DataFrame, required. Must contain labels sample1 and 
        sample2 as columns. 
    @param sample1, sample2: labels in counts_df for which counts will be used 
        as x and y for regplot.
    @param locus_prefix: str, optional. If provided, filter genes by those 
        containing locus_prefix. Intended for filtering inputs to only genes 
        from a given organism. 
    @param xlim, ylim: tuples, optional. If provided, will be used to set x and 
        y axis limits of regplot. 
    @param depth_normalize: bool, default False. If True, normalize counts by 
        sequencing depth of relevant genes (after locus_prefix filtering). Uses
        the greater of the two samples sequencing depths. 
    @param ax: Matplotlib Axes object, optional. If provided, draw plot on 
        existing axes. Otherwise, create new figure.
    @param fisgize: tuple, default (4,4). When generating new figure (i.e. if
        ax not provided), use this as figsize. 
    @param plot_unit: bool, default False. If True, plot 1:1 dashed line.
    @param anno_PCC: bool, default False. If True, calculate Pearson R2 between
        the samples and annotate as text. 
    @param anno_slop: bool, default False. If True, fit OLS linear regression 
        and annotate the slope as text.  

    @return ax: Matplotlib Axes. Contains regression plot. 
    @return pcc: float. Pearson R2 calculated between samples
    @return slope: slope of OLS linear regression between two samples 
    '''
    if not ax: 
        fig,ax = plt.subplots(1,1,figsize=figsize)
    if sample1 not in counts_df or sample2 not in counts_df:
        raise ValueError('One of provided sample labels is not a column in counts_df.')
    #If locus_prefix is provided, filter counts to those genes whose 
    # names contain locus_prefix. 
    if locus_prefix:
        sample1_counts = counts_df.loc[counts_df.index.str.contains(locus_prefix),
                                        sample1]
        sample2_counts = counts_df.loc[counts_df.index.str.contains(locus_prefix),
                                        sample2]
        if len(sample1_counts) == 0 or len(sample2_counts) == 0:
            raise ValueError('Provided locus_prefix does not match any gene identifiers.')
    else:
        sample1_counts = counts_df.loc[:,sample1]
        sample2_counts = counts_df.loc[:,sample2]
    #Normalize sequencing depths of counts, resulting in float counts values for 
    # sample with lower relevant depth 
    # Note this does not work well at low 
    # serquencing coverage if input counts are integers (as expected).  
    if depth_normalize:
        sample1_depth = sample1_counts.sum()
        sample2_depth = sample2_counts.sum()
        #Lower depth sample counts up to higher depth
        if sample1_depth > sample2_depth:
            sample2_counts = sample2_counts * (sample1_depth/sample2_depth)
        else: 
            sample1_counts = sample1_counts * (sample2_depth/sample1_depth)
    #Calculate pcc and slope 
    pcc, pcc_pval = stats.pearsonr(sample1_counts,sample2_counts)
    linregress_results = stats.linregress(sample1_counts,sample2_counts)
    slope = linregress_results[0]
    #Generate regplot 
    ax = sns.regplot(x=sample1_counts,y=sample2_counts,ax=ax) 
    #If axes limits are provided, use them to set x and/or y-axis limits
    if len(xlim) == 2:
        ax.set_xlim(xlim)
    if len(ylim) == 2:
        ax.set_ylim(ylim)
    #Store ymax for annotating text
    xmax, ymax = ax.get_xlim()[1], ax.get_ylim()[1]
    #plot unit line
    if plot_unit:
        ax.plot([0,np.max((xmax,ymax))],
                [0,np.max((xmax,ymax))],
                linestyle='dashed',color='k',linewidth=0.5)
    #Provide text annotations of pcc and slope 
    if anno_PCC:
        ax.text(s="PCC={:.3f}".format(pcc),
                x=xmax*0.02,y=ymax*0.98,ha='left', va='top', color='k',)
    if anno_slope:
        ax.text(s="Slope={:.3f}".format(slope),
                x=xmax*0.02,y=ymax*0.93,ha='left', va='top', color='k',)
    return ax, pcc, slope 

def feature_counts_regplot(feature,counts1,counts2,
                           feature2='',
                           hue=[],palette={},
                           depth_normalize=False,log_transform=False,
                           log_pseudocount=0,ax=None,figsize=(4,4),
                           xlim=(),ylim=(),xlabel='',ylabel='',
                           plot_unit_line=False,plot_unit_interval=0,
                           anno_PCC=False,anno_slope=False):
    """Compare abundance of a single feature between two different
    quantification formats, counts1 and counts2, across all samples. 

    counts1 and counts2 are assumed to have features as rows and sample as 
    columns. 
    """

    #If no axes provided, make new figure using figsize
    if not ax: 
        fig,ax = plt.subplots(1,1,figsize=figsize)
    #If depth_normalize but no xlim and ylim provided, use defaults
    # assuming relative abundance of genome features 
    if depth_normalize and (len(xlim) == 0 or len(ylim)==0):
        xlim = (-6.5,0)
        ylim = (-6.5,0)
    #Depth normalize, if specified
    if depth_normalize:
        counts1_depths = counts1.sum(axis=0)
        counts2_depths = counts2.sum(axis=0)
        #divide all features by sample-level depths
        counts1 = counts1/counts1_depths
        counts2 = counts2/counts2_depths
    #Log transform, if specified
    if log_transform:
        counts1 = np.log10(counts1+log_pseudocount)
        counts2 = np.log10(counts2+log_pseudocount)
    #Get feature data post-normalization/transformation
    feature_counts1 = counts1.loc[feature,:]
    if feature2: #If provided, select feature2 
        feature_counts2 = counts2.loc[feature2,:]
    else:
        feature_counts2 = counts2.loc[feature,:]
    #Calculate pcc and slope 
    pcc, pcc_pval = stats.pearsonr(feature_counts1,feature_counts2)
    linregress_results = stats.linregress(feature_counts1,feature_counts2)
    slope = linregress_results[0]
    #Generate regplot of feature in counts1 and counts2
    if len(hue)==0:
        ax = sns.regplot(x=feature_counts1,y=feature_counts2,ax=ax) 
    else: 
        if len(palette) == 0:
            raise ValueError("Palette must be provided if providing hue values")
        if  isinstance(hue,pd.Series) and \
            (sum(hue.index==feature_counts1.index) < len(feature_counts1) or \
            sum(hue.index==feature_counts2.index) < len(feature_counts2)): 
            warnings.warn("Provided hue values do not share an index with sample counts.")
        #First plot scatter with hue variable 
        ax = sns.scatterplot(x=feature_counts1,y=feature_counts2,
                             hue=hue,palette=palette,ax=ax,zorder=1,
                            linewidth=0
                             )
        #Plot regression line without scatter
        ax = sns.regplot(x=feature_counts1,y=feature_counts2,ax=ax,scatter=False,
                         line_kws={'zorder':2},color='#000000') 
        sns.move_legend(ax,'upper left',bbox_to_anchor=(1,1))
    
    #If axes limits are provided, use them to set x and/or y-axis limits
    if len(xlim) == 2:
        ax.set_xlim(xlim)
    if len(ylim) == 2:
        ax.set_ylim(ylim)
    #If labels are provided, use them
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    #Get current axes limits for unit line and text annos
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    #If ax limits are same, use y-axis ticks to set x-axis ticks
    if xmin==ymin and xmax ==ymax:
        ax.set_xticks(np.arange(np.rint(xmin),np.rint(xmax)+1,1))
        ax.set_yticks(np.arange(np.rint(xmin),np.rint(xmax)+1,1))
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(xmin,xmax)

    #Plot unit line if specified
    if plot_unit_line:
        unit_line = np.array([np.min((xmin,ymin)), np.max((xmax,ymax))])
        ax.plot(unit_line,unit_line,color='#AAAAAA',linestyle='dashed',zorder=0)
        #Use plot unit interval and fill_between to add shaded interval around unit line 
        if plot_unit_interval:
            ax.fill_between(unit_line,unit_line+plot_unit_interval,unit_line-plot_unit_interval,
                            color='#AAAAAA', alpha=0.1,zorder=0)
        
    #Annotate PCC/slope if specified
    if anno_PCC:
        ax.text(s="PCC={:.3f}".format(pcc),
                x=xmin+(xmax-xmin)*0.02,y=ymin+(ymax-ymin)*0.98,
                ha='left', va='top', color='k',)
    if anno_slope:
        ax.text(s="Slope={:.3f}".format(slope),
                x=xmin+(xmax-xmin)*0.02,y=ymin+(ymax-ymin)*0.92,
                ha='left', va='top', color='k',)
    return ax, pcc, slope

def depth_detection_scatterplot(counts_df,locus_prefix='',
                                ax=None,figsize=(3,3),
                                log_transform_depth=True,
                                xlim=(0,8),ylim=(-0.05,1.05),
                                detection_count_thresh=1,
                                metadata=pd.DataFrame(),hue_label='abundance_quantile',
                                palette={},
                                min_depth=0):
    """For a counts profile, calculate per-sample depth and 
    fraction of detection of genes. Intended for use on subsets of genes
    from one genome at a time with locus_prefix. 
    """
    #If no axes object provided, make new figure
    if not ax: 
        fig,ax = plt.subplots(1,1,figsize=figsize)

    #If locus_prefix is provided, subset to genes containing locus_prefix.
    if len(locus_prefix) > 0: 
        counts_df = counts_df.loc[counts_df.index.str.contains(locus_prefix),:]
    depth_detection_df = pd.DataFrame(index=counts_df.columns)
    #Per sample depth
    depth_detection_df['depth'] = counts_df.sum()
    depth_detection_df['detection'] = (counts_df>=detection_count_thresh).astype(int).mean(axis=0)
    if hue_label != 'abundance_quantile':
        if len(metadata) != len(depth_detection_df):
            raise ValueError("Please provide metadata containing specified hue_label.\nmetadata must be indexed on sample identifiers.")
        else: 
            depth_detection_df[hue_label] = metadata[hue_label]
    else: 
        #Use abundance quantile hues:
        #Encode highest matching abundance quantile as a discrete variable in depth_detection_df
        quantiles = np.arange(0,1,0.1)
        for i,q in enumerate(quantiles):
            depth_quantile_cutoff = depth_detection_df[['depth']].quantile(q,axis=0).iloc[0]
            depth_detection_df.loc[depth_detection_df['depth']>=depth_quantile_cutoff,hue_label] = i
        #If no palette provided when using abundance_quantiles, use default cubehelix paeltte
        if not palette:
            palette = sns.color_palette("ch:start=.2,rot=-.3", n_colors=len(quantiles))
        
    #Further data processing - log transform depth if specified (default True)
    if log_transform_depth:
        #lgo10 transform with a pseudocount of 1 
        depth_detection_df['depth'] = np.log10(depth_detection_df['depth']+1) 
    #Further data processing - subset to samples with greater than min_abundance if provided
    if min_depth > 0:
        depth_detection_df = depth_detection_df.loc[depth_detection_df['depth']>min_depth]
    #Scatter plot and return axes
    ax = sns.scatterplot(depth_detection_df,
                x='depth',y='detection',
                hue=hue_label,palette=palette,
                ax=ax,zorder=1
                ) 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    sns.move_legend(ax,'upper left',bbox_to_anchor=(1,1))
    return ax,depth_detection_df



def relative_abundance_barswarmplot(counts_df,sample_md,locus_prefix,x,hue=None,
                                    sampleID_col='SampleID',
                                    bar_palette=MTX_colors.MG02_bar_palette.values(),
                                    swarm_palette=MTX_colors.MG02_point_palette.values(),
                                    dodge=True,figsize=(8,4),order=None,hue_order=None,
                                    bar_alpha=1,markersize=5,ax=None,
                                    show_legend=True):
    '''Generate a barswarmplot where bar/swarm values are the relative abundance of an organism, 
    estimated as the fraction of counts from counts_df corresponding to genes containing locus_prefix. 

    @param counts_df: pd.DataFrame, required. Expected format is genes as index, samples as columns. 
        Values should be counts for each gene in each sample.
    @param sample_md: pd.DataFrame, required. Must contain columns sampleID_col, 
        x, and hue if specified. 
    @locus_prefix: str, required. Must be a substring of gene identifiers with the assumption that 
        only gene identifiers for the organism of interest contain locus_prefix.
    @param x: label in sample_md, required. Determines sample metadata to use for x position 
        grouping of relative abundance data.
    @param hue: label in sample_md, optional. Determines sample metadata to use for color
        grouping of relative abundance data.
    @param sampleID_col: label in sample_md, default 'SampleID'. sample_md must contain 
        all of the values in the columns of counts_df.
    @param bar_palette, swarm_palette: dict, optional. If provided, all values of hue
        in sample_md must be keys.   
    @param dodge,order,hue_order: Passed to seaborn barplot/swarmplot by bar_swarmplot.  
    @param bar_alpha: [0,1], default 1. Transparency for barplot, useful if using same 
        palette for bar_palette and swarm_palette. 
    @param markersize: float, default 5. Radius of markers in points. Passed to swarmplot as size.
    @param ax: matplotlib Axes, default None. If provided plot will be added to that Axes. If 
    @param show_legend: optional, default True. Whether to show or remove legend. 
    
    @return: matplotlib Axes with relative abundance data plotted as a barswarmplot. 
    '''
    RA_data = sample_md.copy()

    #Set index to sampleID_col so indexes aligned when adding in relative abundance later
    if RA_data.index.name != sampleID_col:
        RA_data = RA_data.set_index(sampleID_col)
    locus_prefix_counts = counts_df.loc[counts_df.index.str.contains(locus_prefix),:].sum()
    
    RA_col_label = 'relative_abundance'
    RA_data[RA_col_label] = locus_prefix_counts/counts_df.sum()
    ax = bar_swarmplot(RA_data,x=x,y=RA_col_label,hue=hue,
                                           dodge=dodge,
                                           bar_palette=bar_palette,
                                           swarm_palette=swarm_palette,
                                          figsize=figsize,bar_alpha=bar_alpha,
                                          markersize=markersize,ax=ax)
    return ax, RA_data 

def mcSEED_violin_stripplot(all_results_df,bacteria_info,mcSEED,strains=[],phenotypes=[],
                            logFC_label='coef',pval_label='qval',xlabel='Strain',
                            alpha=0.05,violinplot_geneset='all',ax=None,
                            ylim=(-10,10),markersize=5,figsize=(2,4),
                            linewidth=0,edgecolor='k',
                            legend=False,palette=None):
    '''Generate a violin stripplot of differential expression test results showing mcSEED annotated
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
    @param phenotypes: array-like, optional, default []. List of regular expressions which will be used to 
    match entries in the 'Phenotype' column of mcSEED. Each regular expression's set of matched genes will be 
    plotted as a separate hue (split into significant/non-significant test results) in the resulting plot. 
    Each entry can be either a phenotype code or a '|' joined list of phenotypes which will be used as a single set. 
    @param logFC_label, pval_label: labels in results_df. Will be used as y-axis of violin/strip-plot and to hue genes
    by their significance 
    @param legend: bool, default False. If True, generates a legend where each phenotype is paired with 
    its corresponding color with an additional marker for gray dots corresponding to genes which 
    did not meet significance. 
    @param palette: dict, optional. Must contain integer keys from 1 to the number of 
    phenotypes. If not provided, will automatically generate mapping from phenotypes
    to colors in the seaborn colorblind palette. 
    '''
    #Do not propagate changes to original results DataFrame 
    all_results_df = all_results_df.copy() 
    #Helper variables for supplied lists of strains and phenotypes
    n_strains = len(strains)
    n_phenotypes = len(phenotypes)
    if xlabel == 'Strain':
        xlabel_plot_order_dict = dict(zip(strains,range(len(strains)))) #For maintaining order of violin plots 
    else: 
        #For other xlabels, retain original order of x unique values in all_results_df 
        x_unique_values = all_results_df[xlabel].unique()
        xlabel_plot_order_dict = dict(zip(x_unique_values,range(len(x_unique_values)))) #For maintaining order of violin plots 

    NS_gray = MTX_colors.NS_gray 

    #QC supplied values
    if n_strains == 0:
        raise ValueError('Please supply a list of at least one strain identifiers in strains; currently empty.')
    if not palette: 
        #Set up mcSEED palette based on number of phenotypes provided 
        if n_phenotypes <= 6: #Use ordered paired hues from colorblind_subset
            mcSEED_palette = dict(zip(range(1,7),MTX_colors.colorblind_subset))
        elif n_phenotypes <= 12: #Use paired seaborn palette
            mcSEED_palette = dict(zip(range(1,11),MTX_colors.twelve_paired))
        else:     
            warnings.warn('Number of specified phenotypes exceeds palette length, colors will be repeated.')
            mcSEED_palette = dict(zip(range(1,n_phenotypes+1),sns.color_palette('colorblind',n_phenotypes)))
        mcSEED_palette[0] = NS_gray #Add in no-significance color
    else: 
        mcSEED_palette = palette
    
    
    #Add strain and phenotype metadata to results_df 
    all_results_df['Strain'] = all_results_df.index.str.extract(r'(\w+)_\d+',expand=False).map(bacteria_info['organism'])
    #Use provided list of strains to extract locus tags from results_df corresponding to genes from those organisms
    #Locus prefixes for provided list of strains
    bacteria_lt_prefixes = [bacteria_info[bacteria_info['organism']==strain].index[0] for strain in strains]
    #Extract corresponding genes from all_results_df 
    bacteria_lt_re_pat = '|'.join(bacteria_lt_prefixes)
    bacteria_results_df = all_results_df.loc[all_results_df.index.str.contains(bacteria_lt_re_pat)]
    #Sort by provided list of strains (or list of unique values in col xlabel)
    bacteria_results_df = bacteria_results_df.sort_values(xlabel,key=lambda x:x.map(xlabel_plot_order_dict))
    
    #Filter bacteria_results to bacteria results with mcSEED annotations 
    mcSEED_results_df = bacteria_results_df[bacteria_results_df.index.isin(mcSEED.index)]
    #Encode mcSEED phenotypes from provided list of regexps
    bacteria_results_df['Phenotype'] = 0
    mcSEED_results_df['Phenotype'] = 0
    #stripplot results will contain: 
    #1. only those genes from mcSEED results matching one of the provided phenotype regular expressions
    #2. duplicate entries for locus tags/ DE test results that match multiple of the regular expressions 
    mcSEED_results_for_stripplot = pd.DataFrame(columns=mcSEED_results_df.columns)
    for j,pht in enumerate(phenotypes):
        pht_re = '{0};|{0}$'.format(pht)
        all_pht_loci = mcSEED[mcSEED['Phenotype'].str.contains(pht_re)]
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

    #Handle case where concatenating across models and rownames (genes) have duplicates
    if len(mcSEED_results_for_stripplot.index.unique()) < len(mcSEED_results_for_stripplot):
        mcSEED_results_for_stripplot = mcSEED_results_for_stripplot.reset_index(drop=False)
    if len(violinplot_results.index.unique()) < len(violinplot_results):
        violinplot_results = violinplot_results.reset_index(drop=False)

    #Violin_stripplot call: 
    ax = violin_stripplot(violinplot_results,mcSEED_results_for_stripplot,
                                 logFC_label=logFC_label,
                                pval_label=pval_label,
                                xlabel=xlabel,
                                 hue_label='Phenotype',
                                alpha=alpha,ax=ax,violin_norm='area',
                                 legend=False,palette=mcSEED_palette,ylim=ylim,
                                 markersize=markersize,figsize=figsize,
                                 linewidth=linewidth,edgecolor=edgecolor)
    #Custom legend generation for significance + phenotypes
    if legend:
        #For each phenotype, generate dummy marker (Line2D with o marker and no line) using corresponding 
        #mcSEED_palette color (j+1) 
        legend_elements = [ Line2D([0], [0], marker='o', color='w', label=pht,
                          markerfacecolor=mcSEED_palette[j+1], markersize=6) 
                            for j,pht in enumerate(phenotypes)]
        #Add NS legend element (NS.gray o marker)
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='N.S.',
                          markerfacecolor=NS_gray, markersize=6))
        ax.legend(handles=legend_elements, loc='upper right')
        sns.move_legend(ax,'upper left',bbox_to_anchor=(1,1))
    return ax 