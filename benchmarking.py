#! /usr/bin/env python

"""
benchmarking.py
Utility functions for benchmarking TPR and FPR of MTX differential expression testing 
on Zhang et al. synthetic data and in vitro mock communities. 
Evan Lee
Last update: 04/03/25
"""

###====================================================================================###
# Imports 
###====================================================================================###
#Standard numeric, data, visualization packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os, re, warnings 
from sklearn.metrics import roc_curve, RocCurveDisplay
from statsmodels.stats.multitest import multipletests
#sklearn ROC 

#MTX project specific color palettes 
from MTX_utils import MTX_colors

#Default Color Palettes for heatmap visualization 
default_MTX_TPR_cmap = MTX_colors.MTX_TPR_cmap
default_MTX_FPR_cmap = MTX_colors.MTX_FPR_cmap

###====================================================================================###
### Differential Expresion Testing Results Loading
###====================================================================================###
# Standardized functions for loading and manipulating statistical test results from various methods.  
#Standard synth output directory structure is assumed to be:  
# synth_DE_output (or base_output_dir)
# ├── Zhang_M1
# │ ├── strict
# │ │ ├── null-bug #datasets  
# │ │ ├── ... #datasets 
# │ ├── lenient 
# ├── Zhang_M5 #same structure as Zhang_M1
# ├── Zhang_M6 #same structure as Zhang_M1
# ├── DESeq2
# │ ├── DESeq2_CSS_none
# │ │ ├── null-bug #datasets
# │ │ ├── ... 
# │ ├── DESeq2_TSS_none

# Mock community directory structure is as follows: 
# method_output
# ├── DA_comparisons
# │ ├── DESeq2
# │ │ ├── DESeq2_CSS_none
# │ │ │ ├── Pco0-Pco0 #datasets
#           ...
# │ │ ├── DESeq2_TSS_none
# │ ├── MPRAnalyze
# │ │ ├── MPRAnalyze_TSS_mgx
# │ │ ├── MPRAnalyze_mgx
# │ └── MTXmodel
# │     └── Zhang_M6
# │         ├── lenient
# │         ├── strict

def get_zhang_results_dir(model,dataset,rna_dna_filt='strict',base_output_dir="synth_DE_output"):
    #Standardized directory structure paths for synth DE output for Zhang models
    #See documentation for load_zhang_results 
    if not re.match("Zhang_",model):
        dir_format_model_name = "Zhang_{0}".format(model)
    else:
        dir_format_model_name = model
    zhang_results_dir = os.path.join(base_output_dir,dir_format_model_name,rna_dna_filt,dataset)
    return zhang_results_dir

def get_DESeq_results_dir(model,dataset,base_output_dir="synth_DE_output"):
    #Standardized directory structure for DESeq2 results
    DES_results_dir = os.path.join(base_output_dir,model,dataset)
    return DES_results_dir

def get_MPRAnalyze_results_dir(model,dataset,base_output_dir="synth_DE_output"):
    #Standardized directory structure for MPRAnalyze DE results 
    MPRAnalyze_results_dir = os.path.join(base_output_dir,model,dataset)
    return MPRAnalyze_results_dir

def load_zhang_results_from_fpath(results_fpath,rename_cols={},dropna_policy='retain'):
    #Standardized loading function for zhang formatted results

    #TSV vs CSV parsing
    if re.search(r'\.tsv',results_fpath):
        results_df = pd.read_csv(results_fpath,sep='\t').set_index('feature')
    elif re.search(r'\.csv',results_fpath):
        results_df = pd.read_csv(results_fpath).set_index('feature')
    
    #Rename columns if rename_cols is provided.
    if len(rename_cols) > 0:
        results_df = results_df.rename(columns=rename_cols)
    #Filter rows based on dropna_policy 
    if dropna_policy == 'any' or dropna_policy == 'all':
        results_df = results_df.dropna(axis=0,how=dropna_policy)
    return results_df

def load_zhang_results(model,dataset,
                        rna_dna_filt='strict',
                        base_output_dir="synth_DE_output",
                        results_fname="all_results.tsv",
                        rename_cols={},dropna_policy='retain',
                        filtered_metadata_list='all',
                        invert_reference_direction=False,reference_replacement_dict={},
                        replace_zero_pval=True,zero_pval_value=2.225074e-308,
                        pval_columns=['pval','qval','padj_all']):
    """Load Zhang model DE test results from model and dataset name using standardized directory structure.

    Parameters: 
    @param model: str, required. The name of the Zhang model for which to load statistical test results. 
    Should be of the format 'Zhang_M1' or 'M1'. 
    @param dataset: str, required. The name of the dataset for which to load statistical test results. 
    Zhang datsaset directories are expected to be of the path format {base_output_dir}/{model}/{rna_dna_filt}/{dataset}
    Optional Parameters:
    @ param rna_dna_filt: {'strict','semi-strict','lenient'}, optional. Default 'strict'. 
    Passed to get_zhang_results_dir and specifies directory structure for rna_dna_filt option used when running MTX_model.
    @param base_output_dir: str, default: 'synth_DE_output'. 
    Passed to get_zhang_results_dir. Specifies the path to the parent folder for DE results for different Zhang models. 
    @param results_fname: str, optional. Default 'all_results.tsv'. 
    @param rename_cols: dict, optional. Option which is used to rename DE test results DataFrame columns, 
    such as to standardize column names across different method outputs. 
    Example use: {"padj":"qval","pvalue":"pval"} will rename the columns 'padj' and 'pvalue' to 'qval' and 'pval' 
    in the returned DataFrame. 
    @param: dropna_policy: {'retain','any','all'}, optional. Default 'retain'. If 'any' or 'all' are provided, 
    will respectively filter out rows from test results using pandas dropna. 
    @param filtered_metadata_list: array-like, default 'all'. If provided, filter results to those corresponding to 
    only metadata values in filtered_metadata_list. Useful if MTXmodel was given multiple fixed effects 
    but only some are relevant comparisons. 
    @param invert_reference_direction: bool, default False. Invert reported log fold changes ('coef'). This 
    is desirable if reference groups were not properly configured during MTXmodel run. In this case, it 
    is recommended to also provide reference_replacement_dict, which will replace values in the 'value' column 
    of the result table to reflect the inverted directionality.
    @param reference_replacement_dict: dict, default {}. If invert_reference_direction=True, entries in the
    'value' column of results_df will be replaced. This is purely for keeping track of the direction of 
    comparisons for downstream interpretation. Example usage: reference_replacement_dict={'[Control]':'[Treatment]'}
    """
    zhang_results_dir = get_zhang_results_dir(model,dataset,
                                            rna_dna_filt=rna_dna_filt,base_output_dir=base_output_dir)
    zhang_results_fpath = os.path.join(zhang_results_dir,results_fname)
    results_df = load_zhang_results_from_fpath(zhang_results_fpath,rename_cols=rename_cols,dropna_policy=dropna_policy)
    #Option for filtering metadata comparisons
    if filtered_metadata_list != 'all':
        if len(filtered_metadata_list) > 0:
            results_df = results_df[results_df['metadata'].isin(filtered_metadata_list)]
    #Use options for loading inverted logFC direction and replacing metadata values. 
    if 'coef' in rename_cols:
        coef_label = rename_cols['coef']
    else: 
        coef_label = 'coef'
    if 'qval' in rename_cols:
        qval_label = rename_cols['qval']
    else: 
        qval_label = 'qval'
    if invert_reference_direction:
        results_df[coef_label] = -1*results_df[coef_label]
        for metadata_value in results_df['value'].unique():
            if metadata_value not in reference_replacement_dict:
                warnings.warn("Metadata value {0} is not in provided reference_replacement_dict.".format(metadata_value)+
                    " Note that directionality of coefficients have been inverted but metadata"+
                    " has not been overwritten.")
            else: 
                results_df['value'] = results_df['value'].replace(reference_replacement_dict)
    #0 handling in p-value columns 
    if replace_zero_pval:
        for col in pval_columns:
            if col in results_df:
                results_df[col] = results_df[col].replace(0,zero_pval_value)
            elif col in rename_cols:
                results_df[rename_cols[col]] = results_df[rename_cols[col]].replace(0,zero_pval_value)


    results_df = results_df.sort_values(qval_label)
    return results_df

def load_zhang_taxon_results(model,dataset,
                        rna_dna_filt='strict',
                        base_output_dir="synth_DE_output",
                        results_fname="all_results.tsv",
                        rename_cols={},dropna_policy='retain',
                        padj_all=False,padj_all_method='fdr_bh',
                        invert_reference_direction=False,reference_replacement_dict={},
                        replace_zero_pval=True,zero_pval_value=2.225074e-308,
                        pval_columns=['pval','qval','padj_all']
                        ):
    """Load Zhang model DE test results from model and dataset name using standardized directory structure
    for by-taxon MTXmodel runs. 

    Parameters: 
    @param model: str, required. The name of the Zhang model for which to load statistical test results. 
    Should be of the format 'Zhang_M1_taxon' or 'M1_taxon'. 
    @param dataset: str, required. The name of the dataset for which to load statistical test results. 
    Zhang datsaset directories are expected to be of the path format {base_output_dir}/{model}/{rna_dna_filt}/{dataset}
    The corresponding taxon results are subdirectories of this dataset.
    Optional Parameters:
    @ param rna_dna_filt: {'strict','semi-strict','lenient'}, optional. Default 'strict'. 
    Passed to get_zhang_results_dir and specifies directory structure for rna_dna_filt option used when running MTX_model.
    @param base_output_dir: str, default: 'synth_DE_output'. 
    Passed to get_zhang_results_dir. Specifies the path to the parent folder for DE results for different Zhang models. 
    @param results_fname: str, optional. Default 'all_results.tsv'. 
    @param rename_cols: dict, optional. Option which is used to rename DE test results DataFrame columns, 
    such as to standardize column names across different method outputs. 
    Example use: {"padj":"qval","pvalue":"pval"} will rename the columns 'padj' and 'pvalue' to 'qval' and 'pval' 
    in the returned DataFrame. 
    @param: dropna_policy: {'retain','any','all'}, optional. Default 'retain'. If 'any' or 'all' are provided, 
    will respectively filter out rows from test results using pandas dropna. 
    @param padj_all: bool, default False. Whether or not to apply multiple test correction across all results
    since taxon Zhang results are tested one genome at a time. Generates column 'padj_all' with these adjusted P-values. 
    @param padj_all_method: str, default 'fdr_bh'. Method handle passed to statsmodels multipletests function
    for P-value correction. 
    @param invert_reference_direction: bool, default False. Invert reported log fold changes ('coef'). This 
    is desirable if reference groups were not properly configured during MTXmodel run. In this case, it 
    is recommended to also provide reference_replacement_dict, which will replace values in the 'value' column 
    of the result table to reflect the inverted directionality.
    @param reference_replacement_dict: dict, default {}. If invert_reference_direction=True, entries in the
    'value' column of results_df will be replaced. This is purely for keeping track of the direction of 
    comparisons for downstream interpretation. Example usage: reference_replacement_dict={'[Control]':'[Treatment]'}
    """
    zhang_results_dir = get_zhang_results_dir(model,dataset,
                                            rna_dna_filt=rna_dna_filt,base_output_dir=base_output_dir)
    #Load each taxon's results and concatenate them into all_taxon_results 
    all_taxon_results = pd.DataFrame()
    for taxon in os.listdir(zhang_results_dir):
        if not re.search('^[A-Za-z0-9]+$',taxon): #not expected taxon format 
            print("Skipping non-alphanumeric subdirectory {0} of {1}.".format(taxon,zhang_results_dir))
            continue
        taxon_results_fpath = os.path.join(zhang_results_dir,taxon,results_fname)
        taxon_results = load_zhang_results_from_fpath(taxon_results_fpath,rename_cols=rename_cols,dropna_policy=dropna_policy)
        all_taxon_results = pd.concat((all_taxon_results,taxon_results))
    #Use options for loading inverted logFC direction and replacing metadata values. 
    if invert_reference_direction:
        all_taxon_results['coef'] = -1*all_taxon_results['coef']
        for metadata_value in all_taxon_results['value'].unique():
            if metadata_value not in reference_replacement_dict:
                warnings.warn("Metadata value {0} is not in provided reference_replacement_dict.".format(metadata_value)+
                    " Note that directionality of coefficients have been inverted but metadata"+
                    " has not been overwritten.")
            else: 
                all_taxon_results['value'] = all_taxon_results['value'].replace(reference_replacement_dict)
    #Sort by ascending qval (most to least significant)
    all_taxon_results = all_taxon_results.sort_values('qval')
    if padj_all:
        #multipletests output [1]: pvals_corrected
        all_taxon_results['padj_all'] = multipletests(all_taxon_results['pval'],method=padj_all_method)[1] 
    #Replace 0 p-vals/qvals 
    if replace_zero_pval:
        for col in pval_columns:
            if col in all_taxon_results:
                all_taxon_results[col] = all_taxon_results[col].replace(0,zero_pval_value)
            elif col in rename_cols:
                all_taxon_results[rename_cols[col]] = all_taxon_results[rename_cols[col]].replace(0,zero_pval_value)

    return all_taxon_results

def load_DESeq_results_from_fpath(results_fpath,rename_cols={},dropna_policy='retain'):
    results_df = pd.read_csv(results_fpath,index_col=0,sep='\t')
    if len(rename_cols) > 0:
        results_df = results_df.rename(columns=rename_cols)
    if dropna_policy == 'any' or dropna_policy == 'all':
        results_df = results_df.dropna(axis=0,how=dropna_policy)
    return results_df

def load_DESeq_results(model,dataset,
                        base_output_dir="synth_DE_output",
                        results_fname="all_results.tsv",
                        rename_cols={},dropna_policy='retain',
                        replace_zero_pval=True,zero_pval_value=2.225074e-308,
                        pval_columns=['pvalue','padj','padj_all']):
    """Load DESeq2 DE test results from model and dataset name using standardized directory structure.

    Parameters: 
    @param model: str, required. The name of the DESeq2 model for which to load statistical test results. 
    @param dataset: str, required. The name of the dataset for which to load statistical test results. 
    DESeq2 datsaset directories are expected to be of the path format {base_output_dir}/DESeq2/{model}/{dataset}
    Optional Parameters:
    @param base_output_dir: str, default: 'synth_DE_output'. 
    Passed to get_DESeq_results_dir. Specifies the path to the parent folder for DE results.
    @param results_fname: str, optional. Default 'all_results.tsv'. 
    @param rename_cols: dict, optional. Option which is used to rename DE test results DataFrame columns, 
    such as to standardize column names across different method outputs. 
    Example use: rename_cols={"padj":"qval","pvalue":"pval"} will rename the columns 'padj' and 'pvalue' to 'qval' and 'pval' 
    in the returned DataFrame. 
    @param: dropna_policy: {'retain','any','all'}, optional. Default 'retain'. If 'any' or 'all' are provided, 
    @param replace_zero_pval: bool, default True. If provided, substitute 0s in pval_columns. 
        These result from p-values in DESeq2 which are below the minimum machine value for R. They cause 
        problems for visualizing (e.g. volcano plots) or otherwise log-transforming p-values.
    @param zero_pval_value: float, default 2.225074e*10**-308 (minimum float value for python). 
        If replace_zero_pval=True, replace 0 entries in pval_columns. 
    @param pval_columns: labels in all_results, default ['pvalue','padj']. If 
        replace_zero_pval=True, these columns will have 0s subtituted with zero_pval_value. 
    will respectively filter out rows from test results using pandas dropna. 
    """
    DES_results_dir = get_DESeq_results_dir(model,dataset,base_output_dir=base_output_dir)
    DES_results_fpath = os.path.join(DES_results_dir,results_fname)
    results_df = load_DESeq_results_from_fpath(DES_results_fpath,rename_cols=rename_cols,
                                                dropna_policy=dropna_policy)
    #0 handling in p-value columns 
    if replace_zero_pval:
        for col in pval_columns:
            if col in results_df:
                results_df[col] = results_df[col].replace(0,zero_pval_value)
            elif col in rename_cols:
                results_df[rename_cols[col]] = results_df[rename_cols[col]].replace(0,zero_pval_value)
    return results_df

def load_MPRAnalyze_results_from_fpath(results_fpath,rename_cols={},dropna_policy='retain'):
    #Standardized loading function for MPRAnalyze formatted results

    #TSV vs CSV parsing
    if re.search(r'\.tsv',results_fpath):
        results_df = pd.read_csv(results_fpath,sep='\t',index_col=0)
    elif re.search(r'\.csv',results_fpath):
        results_df = pd.read_csv(results_fpath,index_col=0)
    #Rename columns if rename_cols is provided.
    if len(rename_cols) > 0:
        results_df = results_df.rename(columns=rename_cols)
    #Filter rows based on dropna_policy 
    if dropna_policy == 'any' or dropna_policy == 'all':
        results_df = results_df.dropna(axis=0,how=dropna_policy)
    return results_df

def load_MPRAnalyze_results(model,dataset,
                        base_output_dir="synth_DE_output",
                        results_fname="all_results.tsv",
                        rename_cols={},dropna_policy='retain',
                        replace_zero_pval=True,zero_pval_value=2.225074e-308,
                        pval_columns=['pval','fdr']
                        ):
    """Load MPRAnalyze Comparative Analysis results for model and dataset name using 
    standardized directory structure.

    Parameters: 
    @param model: str, required. The name of the DESeq2 model for which to load statistical test results. 
    @param dataset: str, required. The name of the dataset for which to load statistical test results. 
    MPRAnalyze datsaset directories are expected to be of the path format {base_output_dir}/MPRAnalyze/{model}/{dataset}
    Optional Parameters:
    @param base_output_dir: str, default: 'synth_DE_output'. 
    Passed to get_DESeq_results_dir. Specifies the path to the parent folder for DE results.
    @param results_fname: str, optional. Default 'all_results.tsv'. 
    @param rename_cols: dict, optional. Option which is used to rename DE test results DataFrame columns, 
    such as to standardize column names across different method outputs. 
    Example use: rename_cols={"padj":"qval","pvalue":"pval"} will rename the columns 'padj' and 'pvalue' to 'qval' and 'pval' 
    in the returned DataFrame. 
    @param: dropna_policy: {'retain','any','all'}, optional. Default 'retain'. If 'any' or 'all' are provided, 
    will respectively filter out rows from test results using pandas dropna. 
    @param replace_zero_pval: bool, default True. If True, replace 0 values in pval_columns
        with zero_pval_value (default is minimum python float value)
    @param zero_pval_value: float, default 2.225074e*10**-308 (minimum float value for python). 
        If replace_zero_pval=True, replace 0 entries in pval_columns. 
    @param pval_columns: labels in all_results, default ['pvalue','padj']. If 
        replace_zero_pval=True, these columns will have 0s subtituted with zero_pval_value. 
    """
    MPRAnalyze_results_dir = get_MPRAnalyze_results_dir(model,dataset,base_output_dir=base_output_dir)
    MPRA_results_fpath = os.path.join(MPRAnalyze_results_dir,results_fname)
    results_df = load_MPRAnalyze_results_from_fpath(MPRA_results_fpath,
                        rename_cols=rename_cols,dropna_policy=dropna_policy)
    #0 handling in p-value columns 
    if replace_zero_pval:
        for col in pval_columns:
            if col in results_df:
                results_df[col] = results_df[col].replace(0,zero_pval_value)
            elif col in rename_cols:
                results_df[rename_cols[col]] = results_df[rename_cols[col]].replace(0,zero_pval_value)
    return results_df


###====================================================================================###
### Benchmarking - simulated data (Zhang)
###====================================================================================###

def zhang_significant_spiked_correspondence(all_results, spiked_features=pd.DataFrame(),dataset_name="",
                                            model="",print_output=False,
                                           significant_results={},alpha=0.05,sig_col="qval",
                                            fpr_sig_col="",skip_tpr=False,
                                            include_ppv=True,
                                           check_direction=False,results_sign_col="coef"):
    """Determine basic summary statistics about statistical test results contained in all_results and the set of 
    true simulated DE spiked_features.
    
    Required parameters:
    @param all_results: DataFrame, required: indexed on gene/feature identifiers. Must contain sig_col 
    if significant_results is not provided and results_sign_col if check_direction is set to True (default). 
    
    Optional parameters: 
    @param spiked_features: DataFrame, optional: must contain column 'direction' if check_direction=True.
    which correspond to MTX genes/features in all_results which have spiked-in/ground truth differential expression. 
    @param dataset_name: str, optional. If provided, will fill in appropriate columns in returned summary_df
    @param model: str, optional. If provided, will fill in appropriate columns in returned summary_df
    @param print_output: boolean, default False. If True, will print out calculated TPR and FPR 
    @param significant_results: DataFrame, optional. Default {}. If provided, will skip using alpha to filter 
    all_results into significant_results. Does not support different significance calls for TPR and FPR calculation. 
    @param alpha: float, optional. The threshold cut-off for determining significance based on p-value/q-value columns.
    If not providing pre-calculated significant_results, will be used to filter 
    all_results based on the column(s) specified by tpr_sig_col and fpr_sig_col. Default 0.05. Must be in range (0,1]. 
    @param sig_col: name of variable in all_results, optional. If using alpha to determine significant_results, 
    this must be a column in all_results from which p/q-values <= alpha will be called as significant. 
    @param fpr_sig_col: name of variable in all_results, optional. If provided, a separate column in all_results 
    (e.g. uncorrected P-values) will be used to determine false positive calls for significance. 
    Default empty string.
    @param check_direction: boolean, optional. If provided, true positives must have positive/negative directionality
    in the column specified by results_sign_col corresponding to 'direction' in the spiked_features DataFrame. 
    @param results_sign_col: name of variable in all_results, optional. If check_direction is True, signs of values 
    in this column must correspond with the direction in spiked_features in order for a significant result to be 
    called a true positive. 
    """
    #Set significant_results based on alpha or use provided significant_results 
    if type(significant_results) == dict and (alpha <= 0 or alpha > 1):
        raise ValueError("Provide either significant_results (DataFrame) or alpha (float) significance cut-off within the range (0,1].")
    elif len(significant_results) > 0:
        pass #decide how to handle DataFrame vs array significant results 
    elif alpha > 0 and alpha <= 1: #filter based on alpha cut-off for significance 
        significant_results = all_results[all_results[sig_col]<=alpha]
        if fpr_sig_col: #If fpr_sigf_col is provided, use a different column (e.g. nominal P-values) for significance calls
            #This is in order to accommodate how Zhang et al calculate FPR based on nominal P-values while TPR is based on FDR-adjusted
            #q-vals. 
            fp_significant_results = all_results[all_results[fpr_sig_col]<=alpha]
        else:
            fp_significant_results = significant_results
    else:
        raise ValueError("Provided alpha is outside of (0,1] range. Alternatively, provide significant_results.")   
    #Summary DataFrame initialization
    #Making include_ppv an option to support old behavior in previous scripts 
    if include_ppv:
        summary_df_columns = ["dataset","model","n_tested","n_spiked","TPR","FPR","PPV","NPV"]
    else: 
        summary_df_columns = ["dataset","model","n_tested","n_spiked","TPR","FPR"]
    summary_df = pd.DataFrame(columns=summary_df_columns)
    #Number of tested and spiked features 
    n_tested, n_spiked = len(all_results),len(spiked_features)
    #Sensitivity calculation (TPR)
    if not skip_tpr: #Use skip_tpr for 'null' datasets with no spiked features - assigns np.nan to tpr for clarity
        tp_df = all_results[(all_results.index.isin(spiked_features.index)) &
                           (all_results.index.isin(significant_results.index))] 
        fn_df = all_results[(all_results.index.isin(spiked_features.index)) &
                           ~(all_results.index.isin(significant_results.index))]
        if check_direction:
            if results_sign_col not in all_results or results_sign_col not in tp_df:
                raise ValueError("Provided results_sign_col of {0} is not in all_results. Provide a valid column or set check_direction to False.".format(results_sign_col))
            tp_df = tp_df[np.sign(tp_df[results_sign_col])==spiked_features.loc[tp_df.index,'direction']] #filter based on sign of coefficient
        tpr = len(tp_df)/n_spiked
    else:
        tpr = np.nan
    #Specificity and FPR calculation
    fp_df = all_results[~(all_results.index.isin(spiked_features.index)) &
                       (all_results.index.isin(fp_significant_results.index))] 
    tn_df = all_results[~(all_results.index.isin(spiked_features.index)) &
                       ~(all_results.index.isin(fp_significant_results.index))]
    allneg_df = all_results[~(all_results.index.isin(spiked_features.index))]
    fpr = len(fp_df)/len(allneg_df)

    #Calculate PPV and NPV 
    if not skip_tpr: #For null-XX datasets, skip ppv and npv
    #no spiked features and therefore no tp_df or fn_df. 
        #PPV (precision) calculation: 
        ppv = len(tp_df)/(len(tp_df)+len(fp_df))
        #NPV calculation:
        npv = len(tn_df)/(len(tn_df)+len(fn_df))
    else: #for null datasets, these metrics are undefined. 
        ppv = np.nan 
        npv = np.nan
    #Printed output
    if print_output:
        print("Dataset: {0}".format(dataset_name))
        print("Model: {0}".format(model))
        print("True positive rate: {:.3f}".format(tpr))
        print("False positive rate: {:.3f}".format(fpr))
        print("Number of tested features: {:.0f}".format(n_tested))
        if include_ppv:
            print("Positive predictive value: {:.3f}".format(ppv))
            print("Negative predictive value: {:.3f}".format(npv))
    if include_ppv: #include ppv/npv metrics in output dataframe
        summary_df.loc[0,:] = [dataset_name,model,n_tested,n_spiked,tpr,fpr,ppv,npv]
    else:
        summary_df.loc[0,:] = [dataset_name,model,n_tested,n_spiked,tpr,fpr]
    return summary_df

def mannual_ROC_range(all_results,spiked_features,tpr_col='qval',fpr_col='',
                        alpha_step_size=0.005,log_sample=0,log_sample_num=20,n_features=0,
                        subset_to_tested=True):
    """Manually compute TPR and FPR for DE results for a range of alpha values to generate ROC data, with flexibility of 
    using adjusted vs un-adjusted p-values individually for TPR and FPR.

    This function exists because Zhang et al. use BH-adjusted p-values for their TPR calculation and 
    nominal/unadjusted p-values for their FPR calculation (which makes their GLM statistical test
    perform closer to theoretically optimal FPR of 0.05). The built-in sklearn roc_curve or RocCurveDisplay 
    functions do not support unique calculation of TPR and FPR from adjusted vs. unadjusted p-values, so here we are. 
    TPR is defined as the fraction of true DE features (spiked_features) which have FDR-adjusted p-values <= alpha.
    FPR is defined as the fraction of TESTED non-DE features (all_results[~spiked_features]) which have p-values 
    <= alpha. 
    
    Parameters:
    @param all_results: pd.DataFrame, required. Must contain columns specified by tpr_col and fpr_col (if provided). 
    Must be indexed on feature names. 
    @param spiked_features: pd.DataFrame, required. List of true-positive features. Must be indexed on feature names.
    Optional Parameters:
    @param tpr_col: Must be numeric p-value column in all_results. Default 'qval'. At each alpha sampled, 
    TPR will be defined by the fraction of spiked features which have tpr_col <= alpha.
    @param fpr_col:  Column label in all_results. Optional. If not provided, will use same column as tpr_col. 
    At each alpha sampled, FPR will be defined by the fraction of non-spiked features which have fpr_col <= alpha. 
    @param alpha_step_size: float, optional. default=0.005. The sampling step size for alpha between 0 and 1. 
    @param log_sample: float, optional. default=0. log_sample provides an optional boundary below which alphas will
    be sampled on a logarithmic scale (given the left-skew distribution of p-values that might be observed).
    If not provided, alphas will be evenly linearly sampled from 0 to 1. 
    If log_sample is provided, in addition to the linear sampled alphas, it will generate log_sample_num
    log10-sampled alphas ending at log_sample. 0 will always be included as the first alpha sample to ensure the ROC
    starts at origin. 
    For example, providing log_sample=0.005 (with the defaults alpha_size=0.005 and log_sample_num=20) will 
    generate the following alpha sample: [0,5e-23,5e-22,...5e-4,5e-3,0.01,0.015,...1]
    @param log_sample_num: float, optional. Default=20. If log_sample is provided, this is the number of log-sampled
    alphas to choose. The log-sampled alphas will end at the value specified by log_sample
    @param n_features: int, optional. Default=0. If provided, the total dataset (spiked + nonspiked) size will be taken
    as n_features, with nonspiked being calculated as n_features - len(spiked_features). If it is not provided, n_features 
    will be inferred from len(all_results). Should not ever be less than the length of all_results, otherwise 
    TPR and FPR will no longer be bounded by [0,1].
    @param subset_to_tested: boolean, optional. Default=True. If subset_to_tested, will only consider TPR and FPR
    for the subset of features which are present in all_results. That is, true DE features in spiked_features 
    that are not in all_results will not be considered for TPR calculation. 
    """
    #If fpr_col is not provided, use same column as tpr_col.
    #IF subset_to_tested=True, this THEORETICLALY should give equivalent behavior to sklearn ROC (but does not)
    if not fpr_col: 
        fpr_col = tpr_col
    #If subset_to_tested, only consider TPR and FPR for features which are tested (in all_results)
    if subset_to_tested: 
        spiked_features = spiked_features.loc[spiked_features.index.isin(all_results.index)]
    n_spiked = len(spiked_features) #n_spiked will be used as the number of positive features (TP+FN)
    #
    if not n_features: 
        n_features = len(all_results)
        n_nonspiked = len(all_results.loc[~all_results.index.isin(spiked_features.index)])
    else: #use provided n_features
        #This hasn't been thoroughly tested - results in truncated ROC curves when n_features > len(all_results)
        n_nonspiked = n_features - n_spiked 
        
    #Set-up DataFrame for storing ROC curve values 
    all_results = all_results.copy() #create separate DataFrame copy for internal use of TrueDE column 
    ROC_columns = ["TPR","FPR","alpha"]
    ROC_df = pd.DataFrame(columns=ROC_columns)
    alpha_range = np.arange(log_sample,1+alpha_step_size,alpha_step_size)
    #Add additional alpha sampling points on log-scale 
    if log_sample > 0: 
        #endpoint = False since the value specified by log_sample is already included as the start point for the
        #linear arange sample above 
        log_sample_arange = np.logspace(start=np.log10(log_sample)-log_sample_num,stop=np.log10(log_sample),
                                            num=log_sample_num,base=10,endpoint=False)
        #Prepend 0 and log-sampled alphas to alpha_range 
        alpha_range = np.concatenate(([0],log_sample_arange,alpha_range))
    all_results["TrueDE"] = all_results.index.isin(spiked_features.index)
    # display(all_results)
    for i,alpha in enumerate(alpha_range):
        tpr_sig_calls = all_results[all_results[tpr_col]<=alpha]
        alpha_tpr = tpr_sig_calls["TrueDE"].sum()/n_spiked
        fpr_sig_calls = all_results[all_results[fpr_col]<=alpha]
        alpha_fpr = (~fpr_sig_calls["TrueDE"]).sum()/n_nonspiked
        ROC_df.loc[i,:] = [alpha_tpr,alpha_fpr,alpha]
    return ROC_df

def manual_ROC_AUC(ROC_df,steps_method='steps-post'):
    #Calculate AUC (by fairly low resolution integration) on a manually computed ROC_df. 
    #Will slightly underestimate AUC by virtue of monotonically increasing ROCs. 
    AUC_df = ROC_df[["TPR","FPR","alpha"]] #subset columns 
    auc = 0
    for i in AUC_df.index: 
        #Skip last entry in table - no delta to calculate 
        if i == AUC_df.index[-1]:
            continue
        tpr, fpr = AUC_df.loc[i,"TPR"],AUC_df.loc[i,"FPR"]
        if fpr > 1: 
            break
        next_tpr = np.min((AUC_df.loc[i+1,"TPR"],1))
        next_fpr = np.min((AUC_df.loc[i+1,"FPR"],1))
        window_width = next_fpr-fpr
        #Sloppy integration of rectangle (current TPR) + triangle (slope of ROC)
        if steps_method == 'steps-post': 
            #Window height defined only by current tpr 
            window_area = tpr*window_width 
        elif steps_method == 'default': #corresponds to matplotlib plot default of 
            #diagonal line between two points; window area is 
            #rectangle (from current tpr) + triangle for diagonal line component
            window_area = tpr*window_width + .5*(next_tpr-tpr)*window_width
        auc+=window_area
    return auc 

###====================================================================================###
### Benchmarking - real in vitro datasets 
###====================================================================================###

def define_TP_TN_gene_sets(all_results,locus_prefix_filter='',
                            tp_sig_col='padj',tn_sig_col='padj',
                            logFC_col='log2FoldChange',
                            TP_logFC_threshold=1,TN_logFC_threshold=0.5,
                            alpha=0.05,replace_zero_pval=True,
                            zero_pval_value=2.225074e-308,
                            pval_columns=['pvalue','padj']):
    """From a given differential expression results table, calculate true positive (TP)
    and true negative (TN) gene sets. 

    @param all_results: pd.DataFrame, required. Differential expression results table,
        must contain labels specified by tpr_sig_col, fpr_sig_col, and logFC_col. 
    @param locus_prefix_filter: str, optional. If provided, TP and TN gene sets will 
        only contain genes which contain locus_prefix_filter. 
    @param tpr_sig_col, fpr_sig_col, logFC_col: labels in all_results which will be used 
        to apply significance and fold change thresholds to define TP and TN gene sets. 
    @param TP_logFC_threshold: float, default 1.
    @param TN_logFC_threshold: float, default 0.5.
    @param alpha: float, default 0.05. 
    @param replace_zero_pval: bool, default True. If True, replace 0 values in pval_columns
        with zero_pval_value (default is minimum python float value)
    @param zero_pval_value: float, default 2.225074e*10**-308 (minimum float value for python). 
        If replace_zero_pval=True, replace 0 entries in pval_columns. 
    @param pval_columns: labels in all_results, default ['pvalue','padj']. If 
        replace_zero_pval=True, these columns will have 0s subtituted with zero_pval_value. 

    """
    #Do not modify all_result in place.  
    all_results = all_results.copy()

    #Filter by locus_prefix if provided:
    if locus_prefix_filter:
        all_results = all_results.loc[all_results.index.str.contains(locus_prefix_filter),:]

    #Handle 0 p-values if replace_zero_pval is provided. Uses pval_columns 
    # to determine which columns to replace zero values in .   
    if replace_zero_pval:
        for col in pval_columns: 
            if col in all_results.columns:
                all_results[col] = all_results[col] .replace(0,zero_pval_value)

    #Definte TP and TN gene sets
    #TP: |logFC| > TP_logFC_threshold AND tpr_sig_col <= alpha
    true_pos = all_results[(np.abs(all_results[logFC_col])>TP_logFC_threshold) &
                                (all_results[tp_sig_col]<=alpha)]\
                            .sort_values(logFC_col,ascending=False)
    true_pos['direction'] = np.sign(true_pos[logFC_col])
    #TN: |logFC| < TN_logFC_threshold AND fpr_sig_col > alpha
    true_neg = all_results[(np.abs(all_results[logFC_col])<TN_logFC_threshold) & 
                                (all_results[tn_sig_col]>alpha)]\
                            .sort_values(logFC_col,ascending=False)
    true_neg['direction'] = np.sign(true_neg[logFC_col])
    return true_pos, true_neg


def invitro_benchmarking_summary(all_results, true_sig_features,true_ns_features,
                                            dataset_name="",model="",alpha=0.05,
                                            sig_col="qval",fpr_sig_col="",skip_tpr=False,
                                            include_ppv=True,
                                           check_direction=True,results_sign_col="coef"):
    """Determine basic summary statistics about statistical test results contained in all_results. 
    Requires pre-defined lists of true_sig_features and true_ns_features from which TPR and FPR will be calculated.
    
    Required parameters:
    @param all_results: DataFrame, required: indexed on gene/feature identifiers. Must contain sig_col 
    if significant_results is not provided and results_sign_col if check_direction is set to True (default). 
    @param true_sig_features: indexed on gene/feature identifiers. Must contain column 'direction' if 
    if check_direction is set to True (default). 
    
    Optional parameters: 
    @param dataset_name: str, optional. If provided, will fill in appropriate columns in returned summary_df
    @param model: str, optional. If provided, will fill in appropriate columns in returned summary_df
    @param alpha: float, optional. The threshold cut-off for determining significance in all_results using sig_col 
    and fpr_sig_col. Default 0.05. Must be in range (0,1]. 
    @param sig_col: name of variable in all_results, optional. If using alpha to determine significant_results, 
    this must be a column in all_results from which p/q-values <= alpha will be called as significant. 
    @param fpr_sig_col: name of variable in all_results, optional. If provided, a separate column in all_results 
    (e.g. uncorrected P-values) will be used to determine false positive calls for significance. 
    Default empty string.
    @param include_ppv: bool, default True. If True, provide positive predictive 
        and negative predictivevalue calculations in returned summary DataFrame. 
    @param check_direction: boolean, optional. If provided, true positives must have positive/negative directionality
    in the column specified by results_sign_col corresponding to 'direction' in the spiked_features DataFrame. 
    @param results_sign_col: name of variable in all_results, optional. If check_direction is True, signs of values 
    in this column must correspond with the direction in spiked_features in order for a significant result to be 
    called a true positive. 
    """
    
    #Summary DataFrame initialization
    if include_ppv:
        summary_df_columns = ['dataset','model','n_tested','n_true_sig','n_true_ns','TPR','FPR','PPV','NPV']
    else:
        summary_df_columns = ['dataset','model','n_tested','n_true_sig','n_true_ns','TPR','FPR']
    summary_df = pd.DataFrame(columns=summary_df_columns)
    #Number of tested and spiked features 
    n_tested, n_true_sig,n_true_ns = len(all_results),len(true_sig_features),len(true_ns_features)
    #Sensitivity calculation (TPR)
    if not skip_tpr: #Use skip_tpr for 'null' datasets with no spiked features - assigns np.nan to tpr for clarity
        #TPs: those with sig_col < alpha AND in true_sig_features
        tp_df = all_results[(all_results[sig_col]<=alpha) & \
                            (all_results.index.isin(true_sig_features.index))] 
        fn_df = all_results[(all_results[sig_col]>alpha) & \
                            (all_results.index.isin(true_sig_features.index))]
        if check_direction:
            if results_sign_col not in all_results or results_sign_col not in tp_df:
                raise ValueError("Provided results_sign_col of {0} is not in all_results. Provide a valid column or set check_direction to False.".format(results_sign_col))
            tp_df = tp_df[np.sign(tp_df[results_sign_col])==true_sig_features.loc[tp_df.index,'direction']] #filter based on sign of coefficient
        tpr = len(tp_df)/n_true_sig
    else:
        #if spike_tpr=True, assign np.nan for clarity (vs. a TPR of 0). 
        tpr = np.nan
    #Specificity and FPR calculation
    #Optional: if fpr_sig_col is not provided, use sig_col for FP alpha thresholding 
    if fpr_sig_col == '':
        fpr_sig_col = sig_col
    #FPs: those with sig_col < alpha AND in true_ns_features
    fp_df = all_results[(all_results[fpr_sig_col]<=alpha) & \
                        (all_results.index.isin(true_ns_features.index))] 
    tn_df = all_results[(all_results[fpr_sig_col]>alpha) & \
                        (all_results.index.isin(true_ns_features.index))]
    fpr = len(fp_df)/n_true_ns

    #PPV (precision) calculation: 
    ppv = len(tp_df)/(len(tp_df)+len(fp_df))
    #NPV calculation:
    npv = len(tn_df)/(len(tn_df)+len(fn_df))
    
    #Return summary df with metrics
    if include_ppv:
        summary_df.loc[0,:] = [dataset_name,model,n_tested,n_true_sig,n_true_ns,tpr,fpr,ppv,npv]
    else:
        summary_df.loc[0,:] = [dataset_name,model,n_tested,n_true_sig,n_true_ns,tpr,fpr]
    return summary_df

###====================================================================================###
### Data Visualization   
###====================================================================================###

def pvalue_histogram(results_df,title='',sig_col='qval',hist_color=sns.color_palette("tab10")[0],
                    figsize=(4,4),ax=None,alpha=0.05,plot_uniform_level=True,hue_spiked=False,
                    spiked_features=pd.DataFrame()):
    """Plot a histogram of P-values for differential expression results in results_df. 
    Additional support is provided for separately hue-ing ground truth differentially 
    expressed genes (hue_spiked, spiked_features).  

    @param results_df: pd.DataFrame, required. Must contain 'sig_col' which will be used to 
        generate the histogram. Index should be gene identifiers. 
    @param sig_col: label in results_df, default 'qval'. Values across genes for plotting 
        histogram.
    @param hist_color: color (hexcode str or pyplot/seaborn color). Color of bars in histogram
        or ground truth features if hue_spiked=True. 
    @param figsize: tuple, default (4,4). If no matplotlib Axes provided, size for new figure.
    @param ax: matplotlib Axes: if provided, draw on existing axes; otherwise make new figure. 
    @param alpha: float, default 0.05. Significance level used to calculate expected number of
        features per histogram bucket for uniform P-value histograms. 
        Only used if plot_uniform_level=True.
    @param hue_spiked: bool, default True. If True, requires spiked_features and separately 
        hues bins for genes with ground truth DE (those in spiked_features) vs without DE. 
    @param spiked_features: pd.DataFrame, optional. Required if hue_spiked=True, used to 
        determine hue categories for ground truth DE vs no DE. 
    """
    if not ax: 
        fig, ax = plt.subplots(figsize=figsize)
    pvalues = results_df[sig_col]
    n_tests = len(results_df)
    if hue_spiked:
        if len(spiked_features) == 0:
            warnings.warn('hue_spiked set to True but no spiked_features DataFrame was provided. all bins will have non-significant color.')
        ns_color = MTX_colors.NS_gray
        significant_palette = dict(zip([True,False],[hist_color,ns_color]))
        results_df_sig_annotated = results_df.copy()
        results_df_sig_annotated['spiked'] = results_df_sig_annotated.index.isin(spiked_features.index)
        sns.histplot(results_df_sig_annotated,x=sig_col,hue='spiked',hue_order=[True,False],
            palette=significant_palette,multiple='stack',bins=20,ax=ax)

    else:
        sns.histplot(pvalues,color=hist_color,bins=20)
    if plot_uniform_level:
        plt.plot([0,1],[alpha*n_tests,alpha*n_tests],linestyle='--',color='k')
    plt.title(title)
    return ax 
 

def tpr_fpr_heatmaps(summary_df,x,y,
                    tpr_col='TPR',fpr_col='FPR',
                    ordered_pivot_x=[],ordered_pivot_y=[],
                    figures_dir="figures_pdf",figure_fpath_basename="",
                    tpr_cmap=MTX_colors.MTX_TPR_cmap,
                    fpr_cmap=MTX_colors.MTX_FPR_cmap,
                    tpr_vmax=1,fpr_vmax=1,annot=True,precision_decimals=None,
                    na_facecolor="#BBBBBB",xlabel="",ylabel="",
                    titles=['True Positive Rate','False Positive Rate'],
                    subplot=False,figsize=(8,4)):
    """Generate basic heatmap visualization for true positive rate and false 
        positive rates for synthetic or in vitro (mock community) data.

    Parameters:
    @param summary_df: Required, pd.DataFrame. Must have columns specified by 
        x and y as well as 'TPR' and 'FPR'
    @param x, y: Required, labels in summary_df. Will be used as x and y axes 
        in heatmap. 
    @param tpr_col, fpr_col: labels in summary_df, default 'TPR' and 'FPR'. 
        Columns of summary_df to use for TPR and FPR in heatmaps, can
        supply as arguments if labels differ from those defined in other 
        benchmarking functions.
    @param ordered_pivot_x, ordered_pivot_y: Array-like or pd.Index, optional. 
        Correspond to values in summary_df in columns x and y respectively.
        If provided, will filter/sort entries for x and y using .loc
    @param figures_dir: str, optional, default='figures_pdf'. Path to directory
         where figures will be saved. Created if doesn't exist. 
    @param figure_fpath_basename: str, optional, default=''. Figures will be 
        saved to path {figures_dir}/{figure_fpath_basename}_{TPR/FPR}.pdf
    @param tpr_cmap, fpr_cmap: seaborn ColorMap objects, optional. color maps 
        used for TPR and FPR heatmaps, default to seaborn magma and viridis. 
    @param tpr_vmax, fpr_vmax: float [0,1], default 1. If provided, used as 
        vmax argument in respective heatmaps for TPR and FPR
    @param annot: bool, default True. If True, annotate TPR and FPR values 
        as text over each cell of heatmaps. Passed to sns.heatmap. 
    @param precision_decimals: None or int, default None. If provided, will 
        round to the provided number of decimals for annotated values. 
    @param na_facecolor: str, color hexcode or pyplot color abbreviation. 
        Default="#BBBBBB". Fill in color for na entries in 'TPR' or 'FPR'
        e.g. for TPR in 'null' datasets with no spiked features 
    @param subplot: boolean, optional. Default=False. If true, save TPR 
        and FPR heatmaps as vertically arranged in one condensed figure. 

    """
    #Make copy so no edits passed through to original summary DataFrame. 
    summary_df = summary_df.copy() 
    #Convert TPR and FPR to floats for data type compatibility with seaborn 
    summary_df.loc[:,tpr_col] = summary_df.loc[:,tpr_col].astype('float')
    summary_df.loc[:,fpr_col] = summary_df.loc[:,fpr_col].astype('float')
    if precision_decimals:
        summary_df.loc[:,tpr_col] = np.round(summary_df.loc[:,tpr_col]\
                                    .astype('float'),decimals=precision_decimals)
        summary_df.loc[:,fpr_col] = np.round(summary_df.loc[:,fpr_col]\
                                    .astype('float'),decimals=precision_decimals)
    #Pivot summary_df
    #x and y-axes in heatmap correspond to columns and index in pivot oops :tweak face: 
    # This results in intended behavior based on 
    # the parameter names (i.e. x col label -> x axis) 
    summary_tpr_pivot = summary_df.pivot(index=y,columns=x,values=tpr_col)
    summary_fpr_pivot = summary_df.pivot(index=y,columns=x,values=fpr_col)
    #Reorder rows/ columns of pivot/heatmap if specified 
    if ordered_pivot_x:
        summary_tpr_pivot = summary_tpr_pivot.loc[:,ordered_pivot_x]
        summary_fpr_pivot = summary_fpr_pivot.loc[:,ordered_pivot_x]
    if ordered_pivot_y:
        summary_tpr_pivot = summary_tpr_pivot.loc[ordered_pivot_y,:]
        summary_fpr_pivot = summary_fpr_pivot.loc[ordered_pivot_y,:]
    #Generate figure/axes
    if subplot: #Combine heatmaps into one figure 
        fig1,axes = plt.subplots(2,1,figsize=figsize) 
        ax1, ax2 = axes 
    else: 
        fig1, ax1 = plt.subplots(1,1,figsize=figsize)
        fig2, ax2 = plt.subplots(1,1,figsize=figsize)
    #Heatmap calls for TPR and FPR pivots
    for pivot_df, ax, vmax, cmap,title in zip([summary_tpr_pivot,summary_fpr_pivot],
                                        [ax1,ax2],
                                        [tpr_vmax,fpr_vmax],
                                        [tpr_cmap,fpr_cmap],
                                        titles):
        sns.heatmap(pivot_df,annot=annot,vmin=0,vmax=vmax,cmap=cmap,ax=ax)
        ax.set_facecolor(na_facecolor)
        ax.set_title(title)
        #Custom axes labels if provided:
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
    #Save figure - set up figure directory and use plt.savefig 
    os.makedirs(figures_dir,exist_ok=True)
    #Figure paths depending on if subplots (combined figure) or separate. 
    if subplot:
        heatmap_fpath = os.path.join(figures_dir,"{0}_{1}-{2}.pdf".format(figure_fpath_basename,
                                                                        tpr_col,fpr_col))
        #Delete redundant xtick labels in TPR heatmap
        ax1.set_xlabel("")
        ax1.set_xticklabels([])
        #Save figure 
        fig1.savefig(heatmap_fpath,dpi=300,facecolor='white',bbox_inches='tight')
    else: 
        tpr_heatmap_fpath = os.path.join(figures_dir,"{0}_{1}.pdf".format(figure_fpath_basename,tpr_col))
        fpr_heatmap_fpath = os.path.join(figures_dir,"{0}_{1}.pdf".format(figure_fpath_basename,fpr_col))
        fig1.savefig(tpr_heatmap_fpath,dpi=300,facecolor='white',bbox_inches='tight')
        fig2.savefig(fpr_heatmap_fpath,dpi=300,facecolor='white',bbox_inches='tight')

def ppv_npv_heatmaps(summary_df,x,y,
                    ppv_col='PPV',npv_col='NPV',
                    ordered_pivot_x=[],ordered_pivot_y=[],
                    figures_dir="figures_pdf",figure_fpath_basename="",
                    ppv_cmap=MTX_colors.MTX_PPV_cmap,
                    npv_cmap=MTX_colors.MTX_NPV_cmap,
                    ppv_vmax=1,npv_vmax=1,annot=True,precision_decimals=None,
                    na_facecolor="#BBBBBB",xlabel="",ylabel="",
                    subplot=False,figsize=(8,4)):
    """Generate basic heatmap visualization for positive predictive value 
        and negative predictive value for synthetic or in vitro (mock community) data.
        This basically wraps tpr_fpr_heatmaps but configures default parameters 
        for labels and color palettes to ensure consistency. 

    Parameters:
    @param summary_df: Required, pd.DataFrame. Must have columns specified by 
        x and y as well as 'PPV' and 'NPV'
    @param x, y: Required, labels in summary_df. Will be used as x and y axes 
        in heatmap. 
    @param ppv_col, npv_col: labels in summary_df, default 'PPV' and 'NPV'. 
        Columns of summary_df to use for PPV and NPV in heatmaps, can
        supply as arguments if labels differ from those defined in other 
        benchmarking functions.
    @param ordered_pivot_x, ordered_pivot_y: Array-like or pd.Index, optional. 
        Correspond to values in summary_df in columns x and y respectively.
        If provided, will filter/sort entries for x and y using .loc
    @param figures_dir: str, optional, default='figures_pdf'. Path to directory
         where figures will be saved. Created if doesn't exist. 
    @param figure_fpath_basename: str, optional, default=''. Figures will be 
        saved to path {figures_dir}/{figure_fpath_basename}_{PPV/NPV}.pdf
    @param ppv_cmap, npv_cmap: seaborn ColorMap objects, optional. color maps 
        used for PPV and NPV heatmaps, default to seaborn flare_r and crest_r. 
    @param ppv_vmax, npv_vmax: float [0,1], default 1. If provided, used as 
        vmax argument in respective heatmaps for PPV and NPV. 
    @param annot: bool, default True. If True, annotate PPV and NPV values 
        as text over each cell of heatmaps. Passed to sns.heatmap. 
    @param precision_decimals: None or int, default None. If provided, will 
        round to the provided number of decimals for annotated values. 
    @param na_facecolor: str, color hexcode or pyplot color abbreviation. 
        Default="#BBBBBB". Fill in color for na entries in 'PPV' or 'NPV'
        e.g. for TPR in 'null' datasets with no spiked features 
    @param subplot: boolean, optional. Default=False. If true, save TPR 
        and FPR heatmaps as vertically arranged in one condensed figure. 

    """
    #Call TPR_FPR heatmaps but specify col labels and default palettes for PPV and NPV
    # specific defaults. Pass rest of arguments from ppv_npv_heatmaps. 
    tpr_fpr_heatmaps(summary_df,x,y,
                    ordered_pivot_x=ordered_pivot_x,
                    ordered_pivot_y=ordered_pivot_y,
                    tpr_col=ppv_col,fpr_col=npv_col,
                    figures_dir=figures_dir,
                    figure_fpath_basename=figure_fpath_basename,
                    tpr_cmap=ppv_cmap,
                    fpr_cmap=npv_cmap,
                    tpr_vmax=ppv_vmax,fpr_vmax=npv_vmax,annot=annot,
                    precision_decimals=precision_decimals,
                    na_facecolor=na_facecolor,xlabel=xlabel,ylabel=ylabel,
                    titles=['Positive Predictive Value','Negative Predictive Value'],
                    subplot=subplot,figsize=figsize)


def sklearn_ROC_plot_single_model(all_results,spiked_features,score_col='qval',
                                model="",dataset_name="",ax=None,model_color=sns.color_palette("tab10")[0],
                                line_alpha=1):
    """Use sklearn built-in ROC curve plotting for a given DataFrame of DE test results. 

    Note 1: TPR and FPR are only calculated within the subset of features which are tested (e.g. all_results)
    Note 2: Given how Zhang et al. calculate TPR and FPR respectively from FDR adjusted and nominal/unadjusted 
    p-values, separate evaluation of TPR and FPR requires a separate function (see manual_ROC_plot_single_model). 
    """
    if not ax: 
        fig,ax = plt.subplots(figsize=(6,6))
    roc_df = pd.DataFrame(index=all_results.index,columns=["y_true","y_score"])
    roc_df['y_true'] = all_results.index.isin(spiked_features.index).astype(int)
    roc_df['y_score'] = 1 - all_results[score_col]
    tpr, fpr, thresholds = roc_curve(y_true=roc_df['y_true'],y_score=roc_df['y_score'],pos_label=1)
    RocCurveDisplay.from_predictions(y_true=roc_df['y_true'],y_pred=roc_df['y_score'],
                                    name=model,plot_chance_level=True,ax=ax,color=model_color,
                                    alpha=line_alpha)
    plt.title(dataset_name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.axis("square")
    return roc_df,tpr,fpr,thresholds

def sklearn_ROC_plot_multiple_models(models_results_dict,spiked_features,score_col='qval',
                                    dataset_name="",plot_chance_level=True,palette=[],ax=None):
    """Use sklearn built-in ROC curve plotting for multiple DataFrames of DE test results. 

    Note: Given how Zhang et al. calculate TPR and FPR respectively from FDR adjusted and nominal/unadjusted 
    p-values, separate evaluation of TPR and FPR requires a separate function (see manual_ROC_plot_single_model). 
    """
    if not ax:
        fig,ax = plt.subplots(figsize=(6,6))
    if len(palette) == 0:
        palette = list(sns.color_palette("colorblind",n_colors=len(models_results_dict)))
    for i,model in enumerate(models_results_dict): 
        #Use either default palette or custom palette for color selection for this model
        if type(palette) == dict: 
            try: 
                model_color = palette[model]
            except KeyError:
                raise ValueError("dictionary palette is missing entry for {0}".format(model))
        elif type(palette) == list: #Custom palette 
            model_color = palette[i]

        all_results = models_results_dict[model]
        roc_df = pd.DataFrame(index=all_results.index,columns=["y_true","y_score"])
        roc_df['y_true'] = all_results.index.isin(spiked_features.index).astype(int)
        roc_df['y_score'] = 1 - all_results[score_col]
        tpr, fpr, thresholds = roc_curve(y_true=roc_df['y_true'],y_score=roc_df['y_score'],pos_label=1)
        if plot_chance_level and i==len(models_results_dict)-1: #Only plot chance level once 
            RocCurveDisplay.from_predictions(y_true=roc_df['y_true'],y_pred=roc_df['y_score'],
                                        name=model,plot_chance_level=True,ax=ax,color=model_color)
        else: 
            RocCurveDisplay.from_predictions(y_true=roc_df['y_true'],y_pred=roc_df['y_score'],
                                        name=model,plot_chance_level=False,ax=ax,color=model_color)
    plt.title(dataset_name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.axis("square")
    leg = plt.legend(loc='lower left',bbox_to_anchor=(1,0))
    return roc_df

def manual_ROC_plot_single_model(all_results,spiked_features,model="",dataset_name="",
                                tpr_col='qval',fpr_col='',alpha_step_size=0.005,
                                log_sample=0,log_sample_num=20,n_features=0,
                                model_color=sns.color_palette("tab10")[0],line_alpha=1,
                                plot_chance_level=True,
                                subset_to_tested=True,ax=None,drawstyle='steps-post'):
    """Manual ROC curve plotting for a given DataFrame of DE test results. 
    
    @param 
    Note: Given how Zhang et al. calculate TPR and FPR respectively from FDR adjusted and nominal/unadjusted 
    p-values, separate evaluation of TPR and FPR requires a separate function (this one). 
    """
    if not ax:
        fig,ax = plt.subplots(figsize=(6,6))
    roc_df = mannual_ROC_range(all_results,spiked_features,
                                tpr_col=tpr_col,fpr_col=fpr_col,alpha_step_size=alpha_step_size,
                                log_sample=log_sample,log_sample_num=log_sample_num,n_features=n_features,
                                subset_to_tested=subset_to_tested)
    auc = manual_ROC_AUC(roc_df,steps_method=drawstyle)
    model_auc_handle = "{0} (AUC = {1})".format(model,"{:.2f}".format(auc))
    ax = plt.plot("FPR","TPR",data=roc_df,color=model_color,label=model_auc_handle,alpha=line_alpha,
            drawstyle=drawstyle) #Post-steps style
    #plot alpha = 0.05
    if 0.05 in roc_df["alpha"].unique():
        plt.plot("FPR","TPR",data=roc_df.loc[roc_df["alpha"]==0.05],marker="o",color=model_color,label="")
    if plot_chance_level:
        plt.plot([0,1],[0,1],linestyle='--',color='k',label="Chance level (AUC = 0.5)")
    plt.title(dataset_name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.axis("square")
    leg = plt.legend(loc='lower right')#(loc='lower left',bbox_to_anchor=(1,0))
    return roc_df

def manual_ROC_plot_multiple_models(models_results_dict,spiked_features,dataset_name="",
                                    tpr_col='qval',fpr_col='',alpha_step_size=0.005,
                                log_sample=0,log_sample_num=20,n_features=0,subset_to_tested=True,
                                palette=[],line_alpha=1,plot_chance_level=True,ax=None,
                                drawstyle='steps-post'):
    if not ax: 
        fig,ax = plt.subplots(figsize=(6,6))
    if len(palette) == 0: #Use default palette (seaborn colorblind)
        palette = list(sns.color_palette("colorblind",n_colors=len(models_results_dict)))
    for i,model in enumerate(models_results_dict): 
        #Use either default palette or custom palette for color selection for this model
        if type(palette) == dict: 
            try: 
                model_color = palette[model]
            except KeyError:
                raise ValueError("dictionary palette is missing entry for {0}".format(model))
        elif type(palette) == list: #Custom palette 
            model_color = palette[i]
        all_results = models_results_dict[model]
        roc_df = mannual_ROC_range(all_results,spiked_features,
                                tpr_col=tpr_col,fpr_col=fpr_col,alpha_step_size=alpha_step_size,
                                log_sample=log_sample,log_sample_num=log_sample_num,n_features=n_features,
                                subset_to_tested=subset_to_tested)
        auc = manual_ROC_AUC(roc_df,steps_method=drawstyle)
        model_auc_handle = "{0} (AUC = {1})".format(model,"{:.2f}".format(auc))
        ax = plt.plot("FPR","TPR",data=roc_df,color=model_color,label=model_auc_handle,
                        drawstyle=drawstyle) #Post-steps style plot)
        #plot alpha = 0.05
        if 0.05 in roc_df["alpha"].unique():
            plt.plot("FPR","TPR",data=roc_df.loc[roc_df["alpha"]==0.05],marker="o",color=model_color,label="")
    if plot_chance_level:
        plt.plot([0,1],[0,1],linestyle='--',color='k',label="Chance level (AUC = 0.5)")
    plt.title(dataset_name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.axis("square")
    leg = plt.legend(loc='lower left',bbox_to_anchor=(1,0))
