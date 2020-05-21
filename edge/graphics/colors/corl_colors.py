dark_blue = [31, 120, 180]
light_blue = [166, 206, 227]

light_orange = [253, 205, 172]
light_purple = [203, 213, 232]

green = [115, 163, 72]
orange = [253, 174, 97]
yellow = [227, 198, 52]

optimistic = tuple([c / 256 for c in light_blue])
cautious = tuple([c / 256 for c in dark_blue])
truth = tuple([c / 256 for c in green])

failure = tuple([c / 256 for c in orange])
unviable = tuple([c / 256 for c in yellow])

cmap_var='cividis'
cmap_meas='seismic'
cmap_proba='seismic'