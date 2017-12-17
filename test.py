#!/usr/bin/python3
# coding: utf-8

from math import *

cdf = lambda x: 1-1/sqrt(1.001+0.0160384*x+6.04831e-06*x**2+9.66738e-10*x**3)
g = lambda x: 1-1/sqrt(q+w*x+e*x**2+r*x**3)
q = 1.02530198212461
w = 0.00803171430779825
e = 3.66276019911393e-06
r = 1.83620486904376e-09


f = open('out8', 'r')
f2 = open('out11', 'w')
for l in f:
    x, y = map(float, l.split())
    xx = cdf(x)
    yy = g(y)
    if 2*sqrt(2)/3 < xx + yy: continue
    f2.write(str(xx-yy) + '\n')
