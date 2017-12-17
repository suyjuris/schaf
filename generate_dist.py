#!/usr/bin/python3
# coding: utf-8

# Generates percentiles from gnuplot-readable data

import sys

if not 3 <= len(sys.argv) <= 4:
    print('Usage:\n  %s <input> <output> [<column>]\n\nSplits <input> at spaces, takes the specified column (starting with 0) and writes percentiles to <output>.' % sys.argv[0])

f_inp = sys.argv[1]
f_out = sys.argv[2]
col = int(sys.argv[3]) if 3 < len(sys.argv) else 0

f1 = open(f_inp, 'r')
data = [float(i.split()[col]) for i in f1]
f1.close()

data.sort()
q = [0] * 101
q[0] = data[0]
for i in range(1, len(q)):
    q[i] = data[(i*len(data) + 99)//100 - 1]

f2 = open(f_out, 'w')
for i, j in enumerate(q[:-1]):
    f2.write('%f %f\n' % (i/100, j))
f2.close()
