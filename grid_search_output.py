#!/usr/bin/python3
# coding: utf-8

# Generate some nice graphs from grid_search output

from math import *
from collections import defaultdict
import os
import sys
import subprocess

if len(sys.argv) != 3:
    print('Usage:\n  %s <input> <output-dir>\n\nRead grid search output (just copy-paste the table) from file <input> and generate some nice graphs into <output-dir>.' % (sys.argv[0],))
    sys.exit(2)

columns = 'trai | test | comp | iter | batch |  rate  | decay | a1 | a2 | b1 | b2 | dropout | l2'.split('|')
columns = [i.strip() for i in columns]
col_target = columns.index('comp')
col_ignore = [columns.index('trai'), columns.index('test'), columns.index('iter')]
col_dropno = [i for i, j in enumerate(columns) if len(j) == 2 and j[0] in 'ab']
    
f = open(sys.argv[1], 'r')
output_dir = sys.argv[2]
gnuplot = 'gnuplot'

smooth_samples = 200

datas = defaultdict(list)
foundflag = False
for l in f:
    if not foundflag and not (l[:2] == '0.' or l[:3] == '* 0'): continue
    foundflag = True
    
    if l.startswith('  ') or l.startswith('* '): l = l[2:]
    l = list(map(float, l[:-1].split()))
    assert(len(l) == len(columns))

    val = l[col_target]
    dropout = l[columns.index('dropout')]
    for i, col_val in enumerate(l):
        if i == col_target: continue
        if i in col_ignore: continue
        if i in col_dropno:
            col_val *= dropout
        datas[i].append((col_val, val))

f.seek(0)
raw = f.read()
f.close()

template_main = '''
    set encoding utf8
    set style line 1 dashtype 1              lw 6 lc rgb '#7f0a13' 
    set style line 2 dashtype (22,6)         lw 6 lc rgb '#104354'
    set style line 3 dashtype (20,4,3,4)     lw 6 lc rgb '#217516'
    set style line 4 dashtype (20,4,3,4,3,4) lw 6 lc rgb '#884bab'
    set style circle radius screen 0.02
    set style fill transparent solid 0.15 noborder

    set terminal pdfcairo size 24cm,10cm
    set output '%s'
'''

template_one = '''
    set xrange [%f:%f]
    set yrange [%f:%f]
    set xlabel '%s'
    set ylabel '%s'
    plot '%s' u 1:2 w circle fc '#104354' t '%s', \
         '%s' u 1:2 w line ls 1 t '%s'
'''

os.makedirs(output_dir, exist_ok=True)

f = open(output_dir + '/data_raw.out', 'w')
f.write(raw)
f.close()

plot_file = output_dir + '/tmp_plot.gnuplot'
f_plot = open(plot_file, 'w')

f_plot.write(template_main % (output_dir + '/output.pdf'))

def normal(x, µ, σ):
    return exp(-(x-µ)**2/(2*σ**2)) / sqrt(2*pi*σ**2)

def smooth(d, t, width):
    return (sum(normal(x, t, width)*y for x,y in d)
          / sum(normal(x, t, width)   for x,y in d))

def write_data(data_file, d):
    f_data = open(data_file, 'w')
    d.sort()
    for x,y in d:
        f_data.write(str(x) + ' ' + str(y) + '\n')
    f_data.close()
    
for i in datas:
    data_file  = '%s/tmp_data%02d_%s.out'      % (output_dir, i, columns[i])
    data_file2 = '%s/tmp_data%02d_%s_mova.out' % (output_dir, i, columns[i])
    d = datas[i]
    
    x_min = min(x for x,y in d)
    x_max = max(x for x,y in d)
    y_min = min(y for x,y in d)
    y_max = max(y for x,y in d)

    pad = 0.05
    t1, t2 = 1 + pad, -pad
    x_min_a = t1 * x_min + t2 * x_max
    x_max_a = t2 * x_min + t1 * x_max
    y_min_a = t1 * y_min + t2 * y_max
    y_max_a = t2 * y_min + t1 * y_max

    width = (x_max - x_min) * 0.04
    t_mova = [i / (smooth_samples-1) * (x_max_a - x_min_a) + x_min_a for i in range(smooth_samples)]
    d_mova = [(x, smooth(d, x, width)) for x in t_mova]

    f_plot.write(template_one % (x_min_a, x_max_a, y_min_a, y_max_a, columns[i], 'comp',
                                 data_file, columns[i], data_file2, 'smoothed'))

    write_data(data_file,  d     )
    write_data(data_file2, d_mova)
    
f_plot.close()

subprocess.call([gnuplot, plot_file])
