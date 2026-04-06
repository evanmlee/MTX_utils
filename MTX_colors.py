import seaborn as sns 

###################################################################
#   Functions for manipulating colors
#https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    #Some libraries for handling colors
    import colorsys
    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def desaturate_color(color, amount=0.5):
    """
    Desaturates the given color by multiplying (1-saturation) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    #Some libraries for handling colors
    import colorsys
    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except:
        c = color
    h,l,s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    #Desautrate by amount; amount = 0 -> retain color,
    # higher amount = more gray (lower sat)
    s_desat = s*(1-amount)
    #Ensure [0,1] bounds (only an issue if amount not in [0,1])
    s_desat = max(0,min(1,s*(1-amount)))
    return colorsys.hls_to_rgb(h, l, s_desat)

###################################################################
#				General utility colors and palettes 			  #

#Seaborn documentation on color palettes: 
#https://seaborn.pydata.org/tutorial/color_palettes.html
#Good resource on some built in palettes:
#https://www.practicalpythonfordatascience.com/ap_seaborn_palette

NS_gray = '#BBBBBB'
violinstrip_NS_gray = '#AAAAAA'
medium_gray = '#888888'
S_gray = '#555555'

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
gradient_five_gray_alt = ["#CCCCCC","#AAAAAA","#999999","#666666","#222222"]


five_blue_variant = ["#9FC9EB","#5299C7","#168FFF","#0153DA","#001C96"]
five_red_variant = ["#FEACA7","#DB6565","#FF2804","#BE1900","#911B00"]
six_blue_variant = ["#9FC9EB","#5299C7","#0986C6","#168FFF","#0153DA","#001C96"]
six_red_variant = ["#FEACA7","#DB6565","#DB3932","#FF2804","#BE1900","#911B00"]


MTX_bar_palette = {0:"#FEACA7",1:"#9FC9EB"}
MTX_point_palette = {0:"#FF2804",1:"#3E58A8"}

###################################################################
#		Colors/ palettes related to BG01 - benchmarking datasets  #

#Glycan-up-down-palette
glycan_updown_palette = {-1:'#E2C56E',0:'#BBBBBB',1:'#DA4922'}
#Arabinan-alt
new_arabinan = '#DA4922'
arabinan = '#DA4922'
old_arabinan_alt = '#F7941D'
glucose = '#E2C56E'

MTX_cubehelix_cmap1 = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
MTX_cubehelix_cmap2 = sns.cubehelix_palette(start=1.5, rot=-.5, as_cmap=True)
MTX_cubehelix_cmap3 = sns.cubehelix_palette(start=1.2, rot=.3, as_cmap=True)

#Heatmap palettes for TPR/FPR/PPV/NPV
MTX_TPR_cmap = sns.color_palette("magma", as_cmap=True)
MTX_FPR_cmap = sns.color_palette("viridis", as_cmap=True)

MTX_PPV_cmap = sns.color_palette("flare_r", as_cmap=True)
MTX_NPV_cmap = sns.color_palette("crest", as_cmap=True)

MTX_TPR_soft_cmap = sns.color_palette("rocket", as_cmap=True)
MTX_FPR_soft_cmap = sns.color_palette("mako", as_cmap=True)

#Some alternative mono-chromatic heatmap palettes:
MTX_light_to_red_cmap = sns.color_palette("Reds", as_cmap=True)
MTX_light_to_blue_cmap = sns.color_palette("Blues", as_cmap=True)
MTX_red_light_cmap = sns.color_palette("light:salmon_r", as_cmap=True)
MTX_blue_light_cmap = sns.color_palette("light:b_r", as_cmap=True)


###################################################################
#	 Colors/ palettes related to BG04 - in vitro cross-feeding    #
subset_conditions = ['Arn-Pco','Arn-Mmu','Arn-Pco+Mmu',
                    'Glc-Pco','Glc-Mmu','Glc-Pco+Mmu',
                     'Aos-Pco','Aos-Mmu','Aos-Pco+Mmu']
nine_color_red_blue_gray = [gradient_five_red[1],gradient_five_red[3],gradient_five_red[4],
                        gradient_five_blue[1],gradient_five_blue[3],gradient_five_blue[4],
                          gradient_five_gray[1],gradient_five_gray[3],gradient_five_gray[4]]
BG04_palette = dict(zip(subset_conditions,nine_color_red_blue_gray))
BG04_NC_conditions = ['Arn-negative control','Glc-negative control','Aos-negative control']
BG04_NC_palette = dict(zip(BG04_NC_conditions,
                           [gradient_five_red[0],gradient_five_blue[0],gradient_five_gray[0]]))

########################################################################
#	 Colors/ palettes related to BG05 - simulated noise mock communities #

#Genome depth category colors for MAGs 

#Default crest palette version - not used 
# mag_categories_crest = sns.color_palette("crest", n_colors=4)
# Using a modified version of crest colors with increased hue 
# and luminance variation on the blue end 
mag_categories_crest = ['#75BB92','#459490','#296485','#1F3566']
mag_crest_whitened = [lighten_color(c,amount=0.9) for c in mag_categories_crest]
mag_categories_cubehelix = sns.color_palette("ch:start=.2,rot=-.3", n_colors=4)
#Binary light/dark blue colors 
mag_binary_blue = {0:gradient_five_blue[0],
                    1:gradient_five_blue[4]}

#mock community categorical palettes (orange-gray, yellow-gray)
desats = [0,0.1,0.2,0.3,0.5,0.7,0.8,1]
#Seaborn built-in alternative versions
arabinan_categorical_palette = sns.color_palette("dark:salmon_r", n_colors=8)
glucose_categorical_palette = sns.dark_palette(glucose, reverse=True, n_colors=8)
#Desaturated (color -> gray) version; doesn't achieve great visual separation,
# hence, we use the seaborn builtin analogs above
# arabinan_categorical_palette = [desaturate_color(new_arabinan,d) for d in desats]
# glucose_categorical_palette = [desaturate_color(glucose,d) for d in desats]

###################################################################
#		Colors/ palettes related to MG02 - P. copri mouse models  #

MG02_bar_palette = {"A":"#FEACA7","C":"#D4D4D4","D":"#9FC9EB"}
MG02_point_palette = {"A":"#FF2804","C":"#000000","D":"#3E58A8"}


