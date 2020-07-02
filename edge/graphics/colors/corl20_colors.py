from matplotlib import cm

green = [115, 163, 72]

value_pen_cm = lambda t: cm.RdBu_r(t)
value_con = tuple([c / 256 for c in green])

value_pen_width = .5
value_con_width = 1