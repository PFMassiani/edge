from matplotlib import cm
from matplotlib import colors

dark_blue = [31, 120, 180]
light_blue = [166, 206, 227]

light_orange = [253, 205, 172]
light_purple = [203, 213, 232]

green = [115, 163, 72]
orange = [253, 174, 97]
yellow = [227, 198, 52]

dark_grey = [169, 169, 169]

optimistic = tuple([c / 256 for c in light_blue])
cautious = tuple([c / 256 for c in dark_blue])

# truth = tuple([c / 256 for c in green])
# failure = tuple([c / 256 for c in orange])
# unviable = tuple([c / 256 for c in yellow])
truth = tuple([c / 256 for c in dark_grey])
failure = tuple([c / 256 for c in dark_grey])
unviable = tuple([c / 256 for c in dark_grey])

cmap_var = cm.cividis
cmap_meas = cm.seismic
cmap_proba = cm.seismic

cmap_q_values = cm.RdBu
# cmap_q_values.set_under('gray', alpha=0)
# cmap_q_values.set_over('gray', alpha=0)