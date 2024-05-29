import os, re 
import pandas as pd 
from importlib import resources as impresources
from Bio import SeqIO
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

def _fix_phenotypes_from_pathways(mcSEED):
	"""TODO: finish this - mcSEED annotations are impossible to parse 
	Helper function to fix discrepancies between 'Functional pathway' and 'Phenotype' columns present in 
	P. copri datasets - locus tags which have multiple mcSEED annotations will only have one phenotype value 
	sometimes representing multiple different Functional pathway values. Similarly, some single pathway values 
	have corresponding semicolon separated lists of phenotypes.  

	Requires mcSEED to be indexed on locus tags. 
	"""
	mcSEED_locus_vc = mcSEED.index.value_counts()
	#single entries - should have only one annotation per locus tag and also exclude entries 
	#which have comma separated lists in Phenotype and Functional pathway
	mcSEED_single_entries = mcSEED.loc[(mcSEED_locus_vc[mcSEED.index]==1) & \
										~(mcSEED['Phenotype'].str.contains(';')) &\
										~(mcSEED['Functional pathway'].str.contains(';'))]
	with pd.option_context('display.max_rows',None):
		display(mcSEED_single_entries)
		pass
	
	pathways_with_multiple_phenotypes = []
	single_entry_pathway_phenotype_map = {}
	for pathway in mcSEED_single_entries['Functional pathway'].unique():
		pathway_single_entries = mcSEED_single_entries[mcSEED_single_entries['Functional pathway']==pathway]
		#Functional pathways with one or more associated phenotypes - leave intact (ignore for processing)
		if(len(pathway_single_entries['Phenotype'].unique())) > 1: 
			pathways_with_multiple_phenotypes.append(pathway)
		#Single entry -> single entry - save and use for mapping ambiguous/incorrect pathway/phenotype pairs later 
		else: 
			single_entry_pathway_phenotype_map[pathway] = pathway_single_entries['Phenotype'].iloc[0]
	#Add unambiguous (one to one) semicolon list pathway / phenotype pairs to single_entry_pathway_phenotype_map
	for pht in mcSEED['Phenotype'].unique():
		if pht not in mcSEED_single_entries['Phenotype'].unique() and ';' in pht:
			pht_mcSEED_entries = mcSEED[mcSEED['Phenotype']==pht]
			if len(pht_mcSEED_entries['Functional pathway'].unique())==1:
				pathway = pht_mcSEED_entries['Functional pathway'].iloc[0]
				single_entry_pathway_phenotype_map[pathway] = pht
	# print(pathways_with_multiple_phenotypes)
	# print(single_entry_pathway_phenotype_map)
	#Iterate over pathways in mcSEED 
	# print(len(mcSEED['Functional pathway'].unique()))
	for pathway in mcSEED['Functional pathway'].unique():
		pathway_mcSEED_loci = mcSEED[mcSEED['Functional pathway']==pathway]
		if len(pathway_mcSEED_loci['Phenotype'].unique())>1 and pathway not in pathways_with_multiple_phenotypes:
			if pathway in single_entry_pathway_phenotype_map:
				# print('Single entry phenotype value: {0}'.format(single_entry_pathway_phenotype_map[pathway]))
				# display(pathway_mcSEED_loci)
				pass
	return mcSEED

def load_mcSEED(mcSEED_fpath,index_label='Locus tag',fix_phenotypes=False,deduplicate_locus_tags=False,
				return_pathway_df=False):
	mcSEED = pd.read_csv(mcSEED_fpath)
	if 'Module1' in mcSEED.columns:
		mcSEED = convert_mcSEED_from_module_format(mcSEED,standardize_categories=True,retain_max_phenotype=True)
	if index_label != '':
		mcSEED = mcSEED.set_index(index_label)
	if fix_phenotypes:
		mcSEED = _fix_phenotypes_from_pathways(mcSEED)
		return mcSEED #TODO remove me too enable deduplicated locus tags 

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
	if return_pathway_df:
		pht_pathway_df = pd.DataFrame(columns=["Functional pathway","Functional category"])
		for pht_str in mcSEED["Phenotype"].unique():
			pht_str_matches = mcSEED.loc[mcSEED["Phenotype"]==pht_str]
			assert(len(pht_str_matches["Functional pathway"].unique())==1)
			functional_pathway = pht_str_matches.iloc[0]["Functional pathway"]
			functional_category = pht_str_matches.iloc[0]["Functional category"]
			split_phts = [pht.strip() for pht in pht_str.split(";")]
			split_paths = [cat.strip().title() for cat in functional_pathway.split(";")]
			for pht, cat in zip(split_phts,split_paths):
				pht_pathway_df.loc[pht,"Functional pathway"] = cat
			if len(functional_category.split(";")) == 1:
				pht_pathway_df.loc[split_phts,"Functional category"] = functional_category
		return mcSEED, pht_pathway_df
	else:
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





 