#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt

V = []
Ib = []
Ie = []

with open('trans.csv') as f:
    c = csv.reader(f, delimiter=",")
    next(c)
    for i in range(0,450):
        next(c)
    
    for row in c:
        V += [float(row[0])]
        Ib += [float(row[1])]
        Ie += [-float(row[2])]
        
fig = plt.figure()
ax = plt.subplot(111)

Ic = np.array(Ie) - np.array(Ib)
β = np.array(Ic) / np.array(Ib)

ax.semilogx(Ib, β, 'b.', label="Current gain")
plt.xlabel("Base current (A)")
plt.ylabel("Gain")
ax.legend()
plt.show()
