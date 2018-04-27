#!/usr/bin/python3
# coding: utf-8

# Generate some nice graphs from classify output

from math import *
from collections import defaultdict
import os
import random
import re
import sys
import subprocess

if len(sys.argv) != 4:
    print('Usage:\n  %s <input> <output-dir> <tagfile>\n\nRead classify output (just redirect or tee its stdout into a file) from file <input> and generate some nice graphs into <output-dir>. <tagfile> should contain a classname and a file in each line.' % (sys.argv[0],))
    sys.exit(2)

f = open(sys.argv[1], 'r')
output_dir = sys.argv[2]
gnuplot = 'gnuplot'
max_samples = 500

data = []
for l in f:
    if not re.match('  [- ][01]\.', l): continue

    l = l.split()
    assert(len(l) == 4)
    data.append((float(l[0]), float(l[1]), float(l[2]), l[3].strip()))
    

f.seek(0)
raw = f.read()
f.close()

os.makedirs(output_dir, exist_ok=True)

f = open(output_dir + '/data_raw.out', 'w')
f.write(raw)
f.close()

def lswallow(s, pre):
    return s[len(pre):] if s.startswith(pre) else s

f = open(sys.argv[3], 'r')

classes = {}
classnames = []
for l in f:
    name, fname = l.strip().split()
    f2 = open(fname, 'r')
    repos = {lswallow(i.strip(), 'https://github.com/') for i in f2}
    f2.close()

    d = [i for i in data if i[-1] in repos]
    if d:
        classes[name] = d
        classnames.append(name)

f.close()

template_main = '''
    set encoding utf8
    set style line 1 dashtype 1              lw 6 lc rgb '#7f0a13' 
    set style line 2 dashtype (22,6)         lw 6 lc rgb '#104354'
    set style line 3 dashtype (20,4,3,4)     lw 6 lc rgb '#217516'
    set style line 4 dashtype (20,4,3,4,3,4) lw 6 lc rgb '#884bab'
    set style circle radius screen 0.02

    set terminal pdfcairo size 24cm,10cm
    set output '%s'
    set xrange [%f:%f]
    set yrange [%f:%f]
    set xlabel '%s'
    set ylabel '%s'
    set xtics (%s)
    set key off
'''

template_one = "'%s' u 1:2 w circle fill transparent solid %f noborder fc '#104354'"

plot_file = output_dir + '/tmp_plot.gnuplot'
f_plot = open(plot_file, 'w')

def write_data(data_file, d):
    f_data = open(data_file, 'w')
    d.sort()
    for x,y in d:
        f_data.write(str(x) + ' ' + str(y) + '\n')
    f_data.close()

xtics_s = ', '.join("'%s' %f" % (name, x) for x, name in enumerate(classnames))
f_plot.write(template_main % (output_dir + '/output.pdf', -0.5, len(classnames) - 0.5, -1, 1, 'Classes', 'Rating', xtics_s))

plots_s = []
for x, name in enumerate(classnames):
    vals = [i[2] for i in classes[name]]
    vals.sort()
    if len(vals) > max_samples:
        frac = max_samples / len(vals)
        newvals = []
        t = 0
        for i in vals:
            t += frac
            if t > 1:
                t -= 1
                newvals.append(i)
        vals = newvals

    #points = [(x + random.random()*0.7-0.35, i) for i in vals]

    # Make the randomness a bit less random
    vals_x = [x + i/(len(vals)-1)*0.7-0.35 for i in range(len(vals))]
    random.shuffle(vals_x)
    points = list(zip(vals_x, vals))
    data_file = '%s/tmp_data_%02d_%s.out' % (output_dir, x, name)
    write_data(data_file, points)

    density = 0.05
    plots_s.append(template_one % (data_file, density))

f_plot.write('plot ' + ', '.join(plots_s))
f_plot.close()

subprocess.call([gnuplot, plot_file])
