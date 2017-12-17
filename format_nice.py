#!/usr/bin/python3
# coding: utf-8

# Nice formatting of the tensorflow print output

import sys

s = sys.argv[1]

def tokens(s):
    last = 0
    for i, c in enumerate(s):
        if c in '[] ':
            if last < i:
                yield s[last:i]
            last = i+1
            yield c

count = 0
last = None
for i in tokens(s):
    if i == '[':
        if last == ']':
            print('\n'*(not count) + '\n' + ' '*count, end='')
        print(i, end='')
        count += 1
    elif i == ']':
        print(i, end='')
        count -= 1
    elif i == ' ':
        print(i, end='')
    else:
        print('%5.2f' % float(i), end='')
    last = i
print()
