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

from MTX_utils import bacteria_info
###====================================================================================###
### Data Filtering Functions 
###====================================================================================###
#Helper function - pull expression rows based on absolute/relative abundance in specific samples 
#approach 1: locus tag (species prefix) in specific samplles based on species abundance 
def filter_expr_by_abundance_bin(expr_df,abundance_df,locus_vc_df,abundance_window=(0,0),filt_strains=[],
                                log_transform=False,pseudocount=0.01):
    """Filter expr_df to a long form dataframe containing all locus tags corresponding to samples containing 
    specific source isolates which have abundances within abundance_window. 
    
    @param expr_df: pd.DataFrame, rows = locus tags, columns = sample_ids
    @param abundance_df: pd.DataFrame, rows = samples, columns = strain/bacterial features 
    @param locus_vc_df: pd.DataFrame, indexed on locus tag prefixes for each isolate/strain and with corresponding
    column 'Strain' which contains strain abbreviations corresponding to columns in abundance_df 
    @param abundance_window: length 2 tuple or list, corresponding to upper and lower bounds of abundance (inclusive)
    for which expression data will be returned. 
    default (0,0) does not provide any abundance window filtering.
    @param filt_strains: list/array-like; Default empty list, no strain filtering (e.g. any strain falling within abundance
    window will have its expression data returned). If contains specific strain abbreviations corresponding to 
    abundance_df columns, will only return expression data corresponding to those strains
    
    @return abundance_filt_expr_df: Long form pd.DataFrame with value column 'Expression' and ID columns 'Strain',
    'Sample', and 'target_id'
    """
    #Pick abundance data for sample and strain pairs within abundance_window
    abundance_windowed = pick_abundance_data(abundance_df,abundance_window,filt_strains)
    long_expr_columns = ["Sample","Strain","target_id","Expression"]
    abundance_bin_expr_df = pd.DataFrame(columns=long_expr_columns)
    for strain in abundance_windowed["Strain"].unique():
        locus_tag_md = locus_vc_df[locus_vc_df["Strain"]==strain]
        if "Locus tag" in locus_tag_md.columns:
            locus_tag = locus_tag_md["Locus tag"].iloc[0]
        else:
            locus_tag = locus_tag_md.index[0]
        abundance_windowed_strain = abundance_windowed[abundance_windowed["Strain"]==strain]
        
        aw_strain_samples = abundance_windowed_strain["Sample"]
        strain_abundance_subsetted_expr = expr_df.loc[expr_df.index.str.contains(locus_tag),aw_strain_samples]
        strain_abundance_subsetted_expr = strain_abundance_subsetted_expr.reset_index().rename(columns={"index":"target_id"})
        strain_expr_long = strain_abundance_subsetted_expr.melt(id_vars=["target_id"],value_name="Expression",
                                                               var_name="Sample")
        strain_expr_long["Strain"] = [strain]*len(strain_expr_long)
        strain_expr_long = strain_expr_long[long_expr_columns]
        abundance_bin_expr_df = pd.concat((abundance_bin_expr_df,strain_expr_long))
    if log_transform:
        if log_transform == 10:
            abundance_bin_expr_df["Expression"] = np.log10(abundance_bin_expr_df["Expression"]+pseudocount)
        elif log_transform == 2: 
            abundance_bin_expr_df["Expression"] = np.log2(abundance_bin_expr_df["Expression"]+pseudocount)
        else:
            abundance_bin_expr_df["Expression"] = np.log(abundance_bin_expr_df["Expression"]+pseudocount)
    return abundance_bin_expr_df

def pick_abundance_data(abundance_df, abundance_window=(0,0), filt_strains=[]):
    """Pick abundance data from abundance_df based on strains having abundance within abundance_window. 
    
    @param abundance_df: see filter_expr_by_abundance_bin
    @param abundance_window: see filter_expr_by_abundance_bin
    @param filt_strains: see filter_expr_by_abundance_bin
    
    @return abundance_samples: pd.DataFrame, long-form with value column 'Abundance' and id columns 'Strain'
    and 'Sample'
    
    Abundance window is treated as inclusive min, exclusive max.  
    """
    abundance_df = abundance_df.reset_index().rename(columns={"index":"Sample"})
    #First melt abundance_df into abundance_windowed
    abundance_windowed = pd.melt(abundance_df,id_vars=["Sample"],value_name="Abundance",var_name="Strain")
    #Filter melted abundance 1. by abundance_window 
    if len(abundance_window) != 2:
        raise ValueError("provided abundance window must be two elements long")
    abundance_min, abundance_max = abundance_window
    if abundance_max > abundance_min: #i.e. skip filtering for default abundance_window of (0,0)
        abundance_windowed = abundance_windowed[(abundance_windowed["Abundance"] >= abundance_min) & \
                                        (abundance_windowed["Abundance"] < abundance_max)]
        assert abundance_windowed["Abundance"].max() < abundance_max
        assert abundance_windowed["Abundance"].min() >= abundance_min
    else: 
        warnings.warn("Provided abundance window: {0} will not result in abundance filtering.".format(str(abundance_window)))
    #Filter melted abundance 2. by filt_strains
    if len(filt_strains) > 0: 
        abundance_windowed = abundance_windowed[abundance_windowed["Strain"].isin(filt_strains)]
    return abundance_windowed

def abundance_window_str(abundance_window):
    lower, upper = abundance_window
    lower_str, upper_str = "{:.2E}".format(lower), "{:.2E}".format(upper)
    aw_str = "[{0}, {1})".format(lower_str,upper_str)
    return aw_str

def apply_abundance_windows(data_df,data_abundance_col,abundance_windows):
    """For a given DataFrame with a column specified by data_abundance_col, apply abundance_windows to add an ordinal column "Abundance Window". 

    @params abundance_windows: list or array-like, contains length 2 tuples specifying lower (inclusive) and upper (exclusive) bounds on abundance
    values. Each abundance window will be converted to a string which will be the values in the "Abundance Window" columns.
    """
    for aw in abundance_windows:
        lower, upper = aw
        aw_str = abundance_window_str(aw)
        data_df.loc[data_df[data_abundance_col].apply(lambda x: (x >= lower and x< upper)),"Abundance Window"] = aw_str
    return data_df

def total_RNA_per_strain_per_sample(expr_df,locus_vc_df,abundance_df=pd.DataFrame(),log_transform=False,
                                    sequencing_depth_norm=True):
    """
    For metatranscriptome in expr_df, calculate total RNA expression for each strain in each sample. Return as long form DataFrame.  
    @param expr_df: pd.DataFrame, rows = locus tags, columns = sample_ids
    @param abundance_df: pd.DataFrame, rows = samples, columns = strain/bacterial features 
    @param locus_vc_df: pd.DataFrame, indexed on locus tag prefixes for each isolate/strain and with corresponding
    column 'Strain' which contains strain abbreviations corresponding to columns in abundance_df 
    @param log_transform: {False, 2, 10}, if False (default), no log transformation. If 2 or 10, corresponding total RNA and relative abundances 
    will be log-transformed by the corresponding base 
    """
    #If abundance_df is provided, set include_abundance = True (controls column inclusion and RA extraction from abundance_df)
    if not abundance_df.empty:
        include_abundance = True
    else:
        include_abundance = False 
    if include_abundance:
        RNA_per_sample_columns = ["Strain","Sample","Relative Abundance","Total RNA"]
    else:
        RNA_per_sample_columns = ["Strain","Sample","Total RNA"]
    sequencing_depth_sample_factors = expr_df.sum(axis=0)
    RNA_per_sample = pd.DataFrame(columns=RNA_per_sample_columns)
    for strain in locus_vc_df["Strain"].unique():
        short_df = pd.DataFrame(columns=RNA_per_sample_columns)
        strain_lt = locus_vc_df[locus_vc_df["Strain"]==strain].index[0]
        strain_data = expr_df.loc[expr_df.index.str.contains(strain_lt)]
        strain_totals = strain_data.sum(axis=0)
        if sequencing_depth_norm:
            strain_totals = strain_totals/sequencing_depth_sample_factors
        short_df["Sample"] = strain_totals.index
        short_df["Strain"] = strain
        short_df["Total RNA"] = strain_totals.tolist()
        if include_abundance:
            short_df["Relative Abundance"] = abundance_df[strain].tolist()
        RNA_per_sample = pd.concat((RNA_per_sample,short_df))
    if log_transform: 
        for feature in ["Total RNA", "Relative Abundance"]:
            if feature not in RNA_per_sample:
                continue
            non_zero_feature_data = RNA_per_sample.loc[RNA_per_sample[feature]>0,feature]
            feature_pseudocount = non_zero_feature_data.min() * .5
            print(feature_pseudocount)
            if log_transform == 2: 
                RNA_per_sample[feature] = np.log2(RNA_per_sample[feature]+feature_pseudocount)
            elif log_transform == 10:
                RNA_per_sample[feature] = np.log10(RNA_per_sample[feature]+feature_pseudocount)
    return RNA_per_sample

def gene_summary_statistics(expr_df,locus_vc_df,abundance_df,agg_level="group",sequencing_depth_norm=True,
                            sample_name_group_re="",log_transform=False):
    if sequencing_depth_norm:
        sequencing_depth_sample_factors = expr_df.sum(axis=0)
    if agg_level == "group":
        gene_summary_statistics_cols = ["Strain","Locus tag","Group","Mean Abundance","Mean Expression","Var Expression"]
    elif agg_level == "all":
        gene_summary_statistics_cols = ["Strain","Locus tag","Mean Abundance","Mean Expression","Var Expression"]
    else:
        raise ValueError("agg_level must be 'group' or 'all'.")
    gene_summary_statistics_df = pd.DataFrame(columns=gene_summary_statistics_cols)
    for strain in locus_vc_df["Strain"].unique():
        short_df = pd.DataFrame(columns=gene_summary_statistics_cols)
        strain_lt = locus_vc_df[locus_vc_df["Strain"]==strain].index[0]
        strain_data = expr_df.loc[expr_df.index.str.contains(strain_lt)]
        if agg_level == "group":
            if not sample_name_group_re:
                raise ValueError("Unspecified sample_name_group_re, please provide a regular expression to extract group information from sample names.")
            groups = expr_df.columns.str.extract(sample_name_group_re,expand=False).unique()
            for group in groups:
                group_strain_expr_data = strain_data.loc[:,strain_data.columns.str.contains(group)]
                group_strain_expr_mean = group_strain_expr_data.mean(axis=1)
                group_strain_expr_var = group_strain_expr_data.var(axis=1)
                group_strain_abundance_mean = abundance_df.loc[abundance_df.index.str.contains(group),strain].mean()

                #Fill short_df 
                short_df["Locus tag"] = group_strain_expr_mean.index
                short_df["Mean Expression"] = group_strain_expr_mean.tolist()
                short_df["Var Expression"] = group_strain_expr_var.tolist()
                short_df["Group"] = group
                short_df["Strain"] = strain
                short_df["Mean Abundance"] = group_strain_abundance_mean
                #Concatenate short_df to gene_summary_statistics
                gene_summary_statistics_df = pd.concat((gene_summary_statistics_df,short_df))
        else:
            #TODO implement aggregation across all samples 
            gene_summary_statistics_df = pd.concat((gene_summary_statistics_df,short_df))

    if log_transform: 
        for feature in ["Mean Abundance","Mean Expression","Var Expression"]:
            if feature not in gene_summary_statistics_df:
                continue
            non_zero_feature_data = gene_summary_statistics_df.loc[gene_summary_statistics_df[feature]>0,feature]
            feature_pseudocount = non_zero_feature_data.min() * .5

            if log_transform == 2: 
                gene_summary_statistics_df[feature] = np.log2(gene_summary_statistics_df[feature]+feature_pseudocount)
                print("Log-scale feature pseudocount for {0}".format(feature))
                print(np.log2(feature_pseudocount))
            elif log_transform == 10:
                gene_summary_statistics_df[feature] = np.log10(gene_summary_statistics_df[feature]+feature_pseudocount)
                print("Log-scale feature pseudocount for {0}".format(feature))
                print(np.log10(feature_pseudocount))
    return gene_summary_statistics_df




