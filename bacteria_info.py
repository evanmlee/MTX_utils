import os, re 
import pandas as pd 
from importlib import resources as impresources
from Bio import SeqIO
import matplotlib.pyplot as plt 
import seaborn as sns 
from MTX_utils import MTX_colors
from . import refs

###====================================================================================###
### Load Bacteria Information (locus prefixes, annotations)   
###====================================================================================###

def load_bacteria_info(bacteria_fname="MG02_bacteria_info.csv",sep=','):
	#Loads table of locus tag prefixes and bacteria names
	info_file = (impresources.files(refs) / bacteria_fname)
	bacteria_info = pd.read_csv(info_file,sep=sep,index_col=0)
	# for col in bacteria_info: #Strip whitespace 
		# bacteria_info[col] = bacteria_info[col].str.strip()
	return bacteria_info

def load_mcSEED_phenotype_pathway_info(phenotypes_fpath='mcSEED_phenotype_pathway_codes.csv'):
	#@param phenotypes_fname: Either path to a table containing phenotype/pathway codes or defaults to a filename in 
	#MTX_utils/refs
	if not os.path.exists(phenotypes_fpath):
		phenotypes_fpath = (impresources.files(refs) / phenotypes_fpath)
	if not os.path.exists(phenotypes_fpath):
		raise ValueError("Could not find a file for phenotypes_fpath. Make sure it exists at the specified location, or use default value and file in MTX_utils/refs.")
	mcSEED_phenotype_df = pd.read_csv(phenotypes_fpath,index_col=0)
	mcSEED_phenotype_df['Functional pathway'] = mcSEED_phenotype_df['Functional pathway'].str.capitalize()
	return mcSEED_phenotype_df

def convert_mcSEED_from_module_format(mcSEED,standardize_categories=False,retain_max_phenotype=True):
	#Rename columns which are identical in content between the two formats
	rename_columns = {'Organism':'Isolate name','Locus.tag':'Locus tag',
						'Name':'Protein name','Role':'Protein product','Module1':'Functional category'}
	mcSEED = mcSEED.rename(columns=rename_columns)

	#Generate Functional pathway column from Module 2 entries for Carbohydrate metabolism
	#and Module 3 entries for Amino acid metabolism, Vitamins, cofactors and micronutrients metabolism, Fermentation
	carb_met_index = mcSEED[mcSEED['Functional category'].str.contains('Carbohydrate metabolism')].index
	non_carb_met_index = mcSEED[~(mcSEED.index.isin(carb_met_index))].index
	mcSEED.loc[carb_met_index,'Functional pathway'] = mcSEED.loc[carb_met_index,'Module2']
	mcSEED.loc[non_carb_met_index,'Functional pathway'] = mcSEED.loc[non_carb_met_index,'Module3']

	#Substitute analogous values in 'Functional category'
	func_cat_substitutions = {'Amino acid metabolism':'Amino acids',
								'Fermentation':'Fermentation products',
								'Vitamins, cofactors and micronutrients metabolism':'Vitamins/cofactors',
								'Carbohydrate metabolism':'Carbohydrate utilization'}
	if standardize_categories:
		for cat, standardized_cat in func_cat_substitutions.items():
			mcSEED['Functional category'] = mcSEED['Functional category'].str.replace(cat,standardized_cat)
	#Reorder columns for consistency with Functional category/pathway formatted data
	pathway_format_column_order = ['Isolate name','Locus tag','Protein name','Protein product',
								'Functional category','Functional pathway','Phenotype']
	if retain_max_phenotype:
		pathway_format_column_order.append('MaxPhenotypeValue')
	mcSEED = mcSEED[pathway_format_column_order]
	return mcSEED

def _fix_phenotypes_and_pathways(mcSEED, mcSEED_phenotype_df):
	"""mcSEED annotations for P. copri do not have matched entries for the columns Functional pathway and Phenotype 
	- it's possible for one or both of these columns to have semicolon separated lists 
	that are not matched one to one between these two columns.

	To handle discrepancies, this function generates a list of problem locus tags: 
	#1. Those with a semicolon in the list of phenotypes. In this case, each phenotype will be mapped via 
	mcSEED_phenotype_df to corresponding pathways (if possible) and a semicolon separated list of 
	pathway names will be substituted for the corresponding Functional pathway entry
	#2. Those with semicolon-separated list of functional pathways. Phenotype will be replaced with a 
	semicolon separated list of their corresponding Phenotpye abbreviations
	#3. Those with multiple annotations per locus tag: get flattened list of associated phenotypes across
	all rows, get matching list of pathway entries and overwrite both Phenotype and Functional pathway for 
	all annotations associated with this locus tag.

	Note that this function will not deduplicate locus tags/ collapse annotations across rows into one row for 
	each locus tag, but it will generate a formatted version of mcSEED annotations that should be compatible 
	for doing so.  

	@param mcSEED: pd.DataFrame, required. mcSEED must be indexed on locus tags in order to handle loci with 
	multiple annotations associated with them. 
	@param mcSEED_phenotype_df: pd.DataFrame, required. DataFrame indexed on Phenotype pathway abbreviations 
	(corresponding to entries in Phenotype in mcSEED) and containing Functional pathway and Functional category 
	columns. Used to map entries in these columns in mcSEED to fill in missing values. 
	"""
	mcSEED_locus_vc = mcSEED.index.value_counts()
	#single entries - should have only one annotation per locus tag and also exclude entries 
	#which have comma separated lists in Phenotype and Functional pathway
	#D E B U G M O D E 

	#1. Entries with semicolon lists in Phenotype and only one annotation per locus tag: OK
	multiple_phenotypes = mcSEED.loc[(mcSEED_locus_vc[mcSEED.index]==1) &
									(mcSEED['Phenotype'].str.contains(';'))
										,:]
	for lt in multiple_phenotypes.index: 
		original_pathway_entry = multiple_phenotypes.loc[lt,'Functional pathway']
		#Generate list of phts which are represented in mcSEED_phenotype_df
		phts = [pht for pht in multiple_phenotypes.loc[lt,'Phenotype'].split('; ') if pht in mcSEED_phenotype_df.index]
		if len(phts) == 0:
			#If none of Phenotype values are in mcSEED_phenotype_df, use original pathway entry 
			new_pathway_entry = original_pathway_entry
		else: 
			new_pathway_entry = '; '.join([mcSEED_phenotype_df.loc[pht,'Functional pathway'] for pht in phts])
		multiple_phenotypes.loc[lt,'Functional pathway'] = new_pathway_entry
	
	#2. Entries with semicolon lists in Functional pathway and only one annotation per locus tag: OK
	
	#Select entries with only one annotation and semicolon-separated pathway entry; also mutually exclusive 
	#with multiple_phenotypes (i.e. prioritize modified annotations from multiple_phenotypes in case where
	#both columns have a semicolon-list)
	multiple_pathways = mcSEED.loc[(mcSEED_locus_vc[mcSEED.index]==1) &
									~(mcSEED.index.isin(multiple_phenotypes.index)) &
									(mcSEED['Functional pathway'].str.contains(';'))]
	#Similar to processing of multiple phenotypes above, except for entries with semicolon-separated lists of 
	#'Functional pathway' entries 
	for lt in multiple_pathways.index:
		original_phenotypes = multiple_pathways.loc[lt,'Phenotype']
		pathways = [pw for pw in multiple_pathways.loc[lt,'Functional pathway'].split('; ') \
					if pw in mcSEED_phenotype_df['Functional pathway'].unique()]
		#If cannot find any of pathways in mcSEED_phenotype_df, use original phenotypes for this record
		if len(pathways) == 0:
			new_phenotypes = original_phenotypes
		#Else, get corresponding phenotype abbreviations for each pathway which is in mcSEED_phenotype_df 
		else:
			new_phenotypes = '; '.join([mcSEED_phenotype_df.index[mcSEED_phenotype_df['Functional pathway']==pw][0] \
								for pw in pathways])
		multiple_pathways.loc[lt,'Phenotype'] = new_phenotypes
	
	#3. Those with multiple annotations (i.e. duplicate locus tag entries with possible differences in phenotype 
	#annotations per entry): OK
	#Collect phenotypes across rows and use to generate corresponding list of Functional pathway entries
	#Set values for Phenotype and Functional pathway for all rows with this locus tag 
	multiple_lts = mcSEED.loc[(mcSEED_locus_vc[mcSEED.index]>1)]
	for lt in multiple_lts.index:
		lt_entries = multiple_lts.loc[lt,:]
		#Nested list of split out Phenotype entries associated with this locus tag (i.e. turn list of unique 
		#semicolon-sep strs into list of unique lists)
		lt_phts = [pht_str.split('; ') for pht_str in lt_entries['Phenotype'].unique()]
		#Flatten - https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
		#I'm dumb and can never remember lol
		lt_phts = [pht for phts in lt_phts for pht in phts]
		#Get corresponding pathway entries
		lt_pathways_str = '; '.join([mcSEED_phenotype_df.loc[pht,'Functional pathway'] for pht in lt_phts])
		lt_phts_str = '; '.join(lt_phts)
		multiple_lts.loc[lt,'Phenotype'] = lt_phts_str
		multiple_lts.loc[lt,'Functional pathway'] = lt_pathways_str
	#Get rest of mcSEED annotations (those not covered in three cases above)
	everything_else = mcSEED.loc[~(mcSEED.index.isin(multiple_phenotypes.index)) &\
								~(mcSEED.index.isin(multiple_pathways.index)) &\
								~(mcSEED.index.isin(multiple_lts.index))]
	#Combine all three records tables and sort_index. 
	#Note cannot sort by original mcSEED index order because of duplicate locus tags 
	all_with_modified_annotations = pd.concat((multiple_phenotypes,multiple_pathways,
												multiple_lts,everything_else)).sort_index()
	#Note original order for entries with multiple annotations per locus tag is NOT maintained 
	# display(all_with_modified_annotations.loc[all_with_modified_annotations.index.value_counts()>1,:]) 
	return all_with_modified_annotations

def load_mcSEED(mcSEED_fpath,mcSEED_phenotype_df=pd.DataFrame(),index_label='Locus tag',
				fix_phenotypes_and_pathways=False,deduplicate_locus_tags=False,
				retain_module_format=False):
	"""Load mcSEED annotations into a standardized format from either functional category/pathway or module
	formatted mcSEED annotation tables.

	@param mcSEED_fpath: str, required. File path to load mcSEED annotations from. Supports .csv or .tsv files.  
	@param mcSEED_phenotype_df: pd.DataFrame, default empty DataFrame. Indexed on phenotype abbreviations and contains columns
	Functional pathway and Functional category. This is used when fix_phenotypes_and_pathways=True to 
	add in missing phenotypes/pathways from locus tags with multiple annotations. 
	@param index_label: label, default 'Locus tag'. If provided, sets this column as the index of returned mcSEED DataFrame. 
	@param fix_phenotypes_and_pathways: bool, default False. Use mcSEED_phenotype_df to fill in missing values
	for locus tags with mulitple annotations
	@param deduplicate_locus_tags: bool, default False. Combine multiple annotations for a given locus tag into 
	one row with semicolon-separated entries for Functional category/Functional pathway/Phenotype
	@param retain_module_format: bool, default False. If True, will not reformat Module1/2/3 version of annotations. If True,
	deduplication of locus tags is not supported and annotations will be returned as is.

	"""
	
	#Handle .csv or .tsv formatted annotation paths
	if re.search(r'\.csv',mcSEED_fpath):
		mcSEED = pd.read_csv(mcSEED_fpath)
	elif re.search(r'\.tsv',mcSEED_fpath):
		mcSEED = pd.read_csv(mcSEED_fpath,sep='\t')
	#Automatically convert module format mcSEED tables - see above convert_mcSEED_from_module_format.
	#Can be overriden with retain_module_format=True
	if 'Module1' in mcSEED.columns and not retain_module_format:
		mcSEED = convert_mcSEED_from_module_format(mcSEED,standardize_categories=True,retain_max_phenotype=True)
	elif retain_module_format: 
		#Deduplication of locus tags and other functions not supported for Module format 
		#annotations - return mcSEED as is.
		#Raise warnings for unsupported handling of phenotypes/duplicate locus tags.
		if fix_phenotypes_and_pathways or deduplicate_locus_tags:
			warnings.warn("fix_phenotypes_and_pathways or deduplicate_locus_tags was set to True but these are not supported with retain_module_format.")
		return mcSEED 
	#Column to use as locus tag index 
	if index_label != '':
		mcSEED = mcSEED.set_index(index_label)
	#TODO: decide when to invoke this 
	if fix_phenotypes_and_pathways:
		if len(mcSEED_phenotype_df) == 0:
			raise ValueError("If using fix_phenotypes_and_pathways, mcSEED_phenotype_df must be provided.")
		mcSEED = _fix_phenotypes_and_pathways(mcSEED, mcSEED_phenotype_df)

	#deduplicate_locus_tags: gives a version of mcSEED annotations with only one entry per locus tag
	#multiple functional category/functional pathway/phenotype annotations are combined in this single entry 
	if deduplicate_locus_tags and index_label in ['Locus tag','Locus.tag']:
		mcSEED_unique = pd.DataFrame(columns=mcSEED.columns)
		mcSEED_vc = mcSEED.index.value_counts()
		for mcSEED_locus in mcSEED_vc.index:
			#More than one entry, more than one unique phenotype string associated -> combine entries
			if mcSEED_vc[mcSEED_locus] > 1 and len(mcSEED.loc[mcSEED_locus,'Phenotype'].unique())>1:
				locus_annotations = mcSEED.loc[mcSEED_locus,:]
				# display(locus_annotations) 
				locus_categories, locus_pathways, locus_phts = _get_unique_mcSEED_info_from_locus_annotations(locus_annotations)
				fill_row = locus_annotations.iloc[-1] #(arbitrarily) use last row mcSEED locus annotations as base for deduplicated entry
				for column, locus_entries in zip(['Functional category','Functional pathway','Phenotype'],
												[locus_categories,locus_pathways,locus_phts]):
					fill_row[column] = '; '.join(locus_entries)
				mcSEED_unique.loc[mcSEED_locus,:] = fill_row
			#More than one entry, only one unique phenotype string associated -> just retain last row 
			elif mcSEED_vc[mcSEED_locus] > 1:
				mcSEED_unique.loc[mcSEED_locus,:] = mcSEED.loc[mcSEED_locus,:].iloc[-1,:]
			#Only one entry -> retain only row give by mcSEED.loc[mcSEED_locus,:]
			else: 
				mcSEED_unique.loc[mcSEED_locus,:] = mcSEED.loc[mcSEED_locus,:]
		mcSEED = mcSEED_unique.loc[mcSEED.index.unique()]

	return mcSEED

def _get_unique_mcSEED_info_from_locus_annotations(locus_annotations): 
	#Helper function; extracts unique entries from semicolon-separated lists of Functional 
	#category, Functional Pathway, and Phenotype columns and returns them as lists. 

	#@param locus_annotations: DataFrame containing multiple mcSEED annotations for a single locus. 
	#@return locus_categories, locus_pathways, locus_phts: lists, unique entries extracted from 
	#columns Functional category, Functional Pathway, and Phenotype. 
	locus_categories, locus_pathways, locus_phts = [],[],[] #Use lists to retain order of annotations
	for _, annotation_row in locus_annotations.iterrows(): 
		#Handle new functional category entries, which are not coupled entry for entry as pathways/phenotypes are
		if annotation_row['Functional category'] not in locus_categories: 
			#If semicolon-separated list, iterate over elements and add new entries
			if ';' in annotation_row['Functional category']: 
				for category in annotation_row['Functional category'].split('; '):
					if category not in locus_categories:
						locus_categories.append(category)
			#If not semicolon-separated list, add new entry 
			else:
				locus_categories.append(annotation_row['Functional category'])
		#Handle phenotypes (semicolon separated list)
		if ';' in annotation_row['Phenotype']:
			#Split both pathways and phenotypes into paired lists (since entries are matched 1:1)
			annotation_row_pathways = annotation_row['Functional pathway'].split('; ')
			annotation_row_phts = annotation_row['Phenotype'].split('; ')
			for i, pht in enumerate(annotation_row_phts): 
				#Handle phenotype (single entry) -> add coupled pathway and phenotype if phenotype is new 
				if pht not in locus_phts: 
					locus_pathways.append(annotation_row_pathways[i])
					locus_phts.append(pht)
		#Handle phenotype (single entry) -> add coupled pathway and phenotype if phenotype is new 
		else:
			if annotation_row['Phenotype'] not in locus_phts: #Add new phenotype and pathway
				locus_pathways.append(annotation_row['Functional pathway'])
				locus_phts.append(annotation_row['Phenotype'])
	return locus_categories, locus_pathways, locus_phts

###====================================================================================###
### Functions utilizing bacterial genome .ffn (rRNA annotations, coding sequences)
###====================================================================================###

def load_bacteria_genome_from_ffn(genome_fname,genomes_dir='reference_genomes'):
	"""Uses BioPython SeqIO to parse a genome fasta into a DataFrame indexed on locus tags
	and containing name, description, and sequence information. 
	@param genome_fname: 
	@param genomes_dir: if genome_fname is not in current path, searches for an directory specified by 
	genomes_dir in both path and refs/

	@return genome_locus_df: pd.DataFrame, indexed on locus tags and containing columns 'name','description'
	and 'sequence' containing corresponding information from each SeqIO.Record object 
	"""
	#Try searching for just genome_fname in path
	if os.path.exists(genome_fname):
		genome_fpath = genome_fname
	#If fails, check for existence of genomes_dir in path 
	elif os.path.exists(genomes_dir):
		genome_fpath = os.path.join(genomes_dir,genome_fname)
		if not os.path.exists(genome_fpath):
			raise ValueError("Could not find genome file specified by {0}/{1}".format(genomes_dir,genome_fname)) 
	#Search in MTX_utils refs
	elif os.path.exists(impresources.files(refs) / genomes_dir):
		genome_fpath = (impresources.files(refs) / genomes_dir / genome_fname)
		print('Using genome from MTX_utils reference_genomes directory.')
		print("File path: {0}".format(genome_fpath))
	else: 
		raise ValueError('Could not find genomes_dir {0} in path or MTX_utils refs.'.format(genomes_dir))
		#genome_fpath = os.path.join(genomes_dir,genome_fname)
	#SeqIO parsing of fasta file stored at genome_fpath
	genome_ffn_dict = SeqIO.to_dict(SeqIO.parse(genome_fpath,'fasta'))
	locus_df_columns = ['name','description','sequence']
	genome_locus_df = pd.DataFrame(columns=locus_df_columns)
	for locus_tag in genome_ffn_dict:
		lt_record = genome_ffn_dict[locus_tag]
		genome_locus_df.loc[locus_tag] = dict(zip(locus_df_columns,
			[lt_record.name,lt_record.description.replace(locus_tag,'').strip(),str(lt_record.seq)]))
	return genome_locus_df

def filter_rRNA_loci(counts_df,genome_fname,genomes_dir='reference_genomes'):
	genome_df = load_bacteria_genome_from_ffn(genome_fname,genomes_dir=genomes_dir)
	rRNA_loci = genome_df.loc[genome_df['description'].str.contains('ribosomal RNA')].index
	rRNA_loci_in_counts_df = rRNA_loci[rRNA_loci.isin(counts_df.index)]
	filtered_counts_df = counts_df.drop(index=rRNA_loci_in_counts_df)
	if len(filtered_counts_df) == len(counts_df):
		warnings.warn('rRNA filtering did not remove any loci from counts_df. Check inputs.')
	return filtered_counts_df

def get_all_rRNA_loci(genomes_dir,rRNA_description_re_pat='ribosomal RNA'):
	genome_fnames = os.listdir(genomes_dir)
	#Remove hidden files (e.g. .DS_store)
	genome_fnames = [fname for fname in genome_fnames if not re.search(r'^\.',fname)]
	all_rRNA_loci_df = pd.DataFrame(columns=['name','description','sequence'])
	for genome_fname in genome_fnames:
		genome_df = load_bacteria_genome_from_ffn(genome_fname,genomes_dir=genomes_dir)
		rRNA_loci_df = genome_df.loc[genome_df['description'].str.contains(rRNA_description_re_pat,regex=True)]
		all_rRNA_loci_df = pd.concat((all_rRNA_loci_df,rRNA_loci_df))
	return all_rRNA_loci_df

def filter_rRNA_loci_all_genomes(counts_df,genomes_dir='',rRNA_description_re_pat='ribosomal RNA',
								all_rRNA_loci_df=pd.DataFrame()):
	"""Remove all loci corresponding to ribosomal RNA from a counts DataFrame.

	@param: counts_df: pd.DataFrame, required. Must be indexed on locus tags/ gene identifiers. 
	@param: genomes_dir: str, optional. Directory containing fasta files of reference genome locus tags. The description
	field of these fasta files will be searched for rRNA_description_re_pat and loci matching will be removed from counts_df.
	@param: all_rRNA_loci_df: pd.DataFrame, optional. Alternative to genomes_dir, provide DataFrame consisting of 
	rRNA loci. Must be indexed on locus tags/ gene identifiers. 
	If both genomes_dir and rRNA_loci_df are provided, genomes_dir will be used to generate the list of rRNA loci to filter. 
	If neither are provided, will raise a ValueError. 
	"""
	if len(all_rRNA_loci_df)==0 and not genomes_dir:
		raise ValueError("Must provide either genomes_dir (file path) or rRNA_loci_df (DataFrame indexed on locus tags).")
	if genomes_dir: #If providing genomes_dir, generate list of rRNA loci using get_all_RNA_loci
		all_rRNA_loci_df = get_all_rRNA_loci(genomes_dir)
	elif isinstance(all_rRNA_loci_df,pd.DataFrame) and len(all_rRNA_loci_df) >= 0: #If providing all_RNA_loci_df
		pass
	all_rRNA_loci = all_rRNA_loci_df.index
	all_rRNA_loci_in_counts_df = all_rRNA_loci[all_rRNA_loci.isin(counts_df.index)]
	filtered_counts_df = counts_df.drop(index=all_rRNA_loci_in_counts_df)
	return filtered_counts_df


###====================================================================================###
### Data Visualization functions for bacterial annotations 
###====================================================================================###

def mcSEED_GSEA_heatmap(mcSEED_GSEA_df,pathway_col='pathway',organism_col='organism',NES_col='NES',
	pval_col='padj',subset_pathways=[],subset_organisms=[],alpha=0.05,cmap_str='RdBu_r',vmin=-2,vmax=2):
	"""Generate a heatmap encoding GSEA Normalized Enrichment Score (NES) and P-value information for 
	mcSEED GSEA results with individual organisms as columns and pathways as rows. 

	@param mcSEED_GSEA_df: pd.DataFrame, required. Must contain labels specified by pathway_col and 
	organism_col which will be used to organize pathway results by organism in the heatmap. 
	@param pathway_col: label in mcSEED_GSEA_df, default 'pathway'. Column for which organism 
	GSEA pathway results are from. 
	@param oragnism_col: label in mcSEED_GSEA_df, default 'organism'. Column for which organism 
	GSEA pathway results are from. 
	@param NES_col: label in mcSEED_GSEA_df for Normalized enrichment scores. Default 'NES'.
	@param pval_col: label in mcSEED_GSEA_df for P-values to use for significance cutoff. Default 'padj'.
	If '' is provided, will ignore significance thresholding and visualize all NES provided regardless of significance.  
	@param subset_pathways, subset_organisms: list or array-like, optional. If provided, heatmap will only contain a subset of 
	tested mcSEED pathways or organisms. 
	@param alpha: float, default 0.05. Significance threshold to be applied to pval_col. Any pathway 
	GSEA results with a p-value > alpha will be converted to np.nan and not colored in heatmap. 
	@param cmap_str: str, default 'RdBu_r'. Must specify a valid seaborn color palette which will be converted 
	into a color map. See: https://seaborn.pydata.org/tutorial/color_palettes
	@param vmin, vmax: float, optional. Passed to sns.heatmap. 
	"""
	#Generate copy to modify 
	heatmap_GSEA_data = mcSEED_GSEA_df.copy().reset_index()
	cmap = sns.color_palette(cmap_str,as_cmap=True)
	if len(subset_pathways) > 0:
		#Drop entries in subset pathways which are not represented in heatmap_GSEA_data 
		subset_pathways = [pathway for pathway in subset_pathways if pathway in heatmap_GSEA_data['pathway'].unique()]
		heatmap_GSEA_data = heatmap_GSEA_data.loc[heatmap_GSEA_data['pathway'].isin(subset_pathways)]
	if len(subset_organisms) > 0:
		heatmap_GSEA_data = heatmap_GSEA_data.loc[heatmap_GSEA_data[organism_col].isin(subset_organisms)]
	if pval_col:
		#Default behavior: use pval_col to threshold NES scores. Anything with a P-value > alpha will be 
		#converted to np.nan and not be visualized 
		heatmap_GSEA_data['NES_sig'] = heatmap_GSEA_data.loc[heatmap_GSEA_data[pval_col]<=alpha,NES_col]
	else:
		#If empty str provided for pval_col, then use unthresholded NES for visualization
		heatmap_GSEA_data['NES_sig'] = heatmap_GSEA_data['NES_sig']
	heatmap_GSEA_data_2D = heatmap_GSEA_data.pivot(index=pathway_col,columns=organism_col,values='NES_sig') #TODO finalize 
	if len(subset_pathways) > 0:
		heatmap_GSEA_data_2D = heatmap_GSEA_data_2D.loc[subset_pathways,:]
	if len(subset_organisms) > 0:
		heatmap_GSEA_data_2D = heatmap_GSEA_data_2D.loc[:,subset_organisms]
	fig,ax = plt.subplots(1,1,figsize=(6,6))
	ax = sns.heatmap(heatmap_GSEA_data_2D,cmap=cmap,cbar=True,
		vmin=vmin,vmax=vmax,xticklabels=True,yticklabels=True,linecolor='#000000',linewidths=0.5)
	ax.set_facecolor(MTX_colors.NS_gray)
	return ax 







 