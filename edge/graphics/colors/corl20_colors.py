from matplotlib import cm

dark_blue = [31, 120, 180]
light_blue = [166, 206, 227]

light_orange = [253, 205, 172]
light_purple = [203, 213, 232]

green = [115, 163, 72]
orange = [253, 174, 97]
yellow = [227, 198, 52]

value_pen_cm = lambda t: cm.RdBu_r(t)
value_con = tuple([c / 256 for c in green])

value_pen_width = .5
value_con_width = 1

def episodic_failure_colors(t):
    return cm.hsv(t)