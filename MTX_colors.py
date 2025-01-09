import seaborn as sns 

#Seaborn documentation on color palettes: 
#https://seaborn.pydata.org/tutorial/color_palettes.html
#Good resource on some built in palettes:
#https://www.practicalpythonfordatascience.com/ap_seaborn_palette

NS_gray = '#BBBBBB'
S_gray = '#888888'

MG02_bar_palette = {"A":"#FEACA7","C":"#D4D4D4","D":"#9FC9EB"}
MG02_point_palette = {"A":"#FF2804","C":"#000000","D":"#3E58A8"}

#seaborn colorblind palette subset 
ten_colorblind = sns.color_palette('colorblind')
#seaborn paired palette 
twelve_paired = sns.color_palette('Paired')

#subset alternates dark, light for blue, red-orange, pink
colorblind_subset = [ten_colorblind[0],ten_colorblind[-1],ten_colorblind[3],
					ten_colorblind[1],ten_colorblind[4],ten_colorblind[6]]
#Five color Adobe gradient palettes
gradient_five_red = ['#FFB3AF','#FF7A6E','#EB321B','#BE1900','#821800']
gradient_five_blue = ['#9FC9EB','#62ABF0','#147EE0','#0059B8','#001987']
gradient_five_green = ["#4DFF97","#00FF6A","#00CC55","#189E50","#00712F"]
gradient_five_gray = ["#DEDEDE","#C4C4C4","#A1A1A1","#787878","#444444"]


five_blue_variant = ["#9FC9EB","#5299C7","#168FFF","#0153DA","#001C96"]
five_red_variant = ["#FEACA7","#DB6565","#FF2804","#BE1900","#911B00"]
six_blue_variant = ["#9FC9EB","#5299C7","#0986C6","#168FFF","#0153DA","#001C96"]
six_red_variant = ["#FEACA7","#DB6565","#DB3932","#FF2804","#BE1900","#911B00"]


MTX_bar_palette = {0:"#FEACA7",1:"#9FC9EB"}
MTX_point_palette = {0:"#FF2804",1:"#3E58A8"}

#Glycan-up-down-palette
glycan_updown_palette = {-1:'#E2C56E',0:'#BBBBBB',1:'#DA4922'}
#Arabinan-alt
new_arabinan = '#DA4922'
old_arabinan_alt = '#F7941D'
glucose = '#E2C56E'

MTX_cubehelix_cmap1 = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
MTX_cubehelix_cmap2 = sns.cubehelix_palette(start=1.5, rot=-.5, as_cmap=True)
MTX_cubehelix_cmap3 = sns.cubehelix_palette(start=1.2, rot=.3, as_cmap=True)

MTX_TPR_cmap = sns.color_palette("magma", as_cmap=True)
MTX_FPR_cmap = sns.color_palette("viridis", as_cmap=True)

MTX_TPR_soft_cmap = sns.color_palette("rocket", as_cmap=True)
MTX_FPR_soft_cmap = sns.color_palette("mako", as_cmap=True)