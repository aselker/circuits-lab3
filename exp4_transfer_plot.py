#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

vals = ["exp3_10k","exp4_20k", "exp4_30k", "exp4_40k"]

vb_exp = {}
ib_exp = {}
ve_exp = {}

for val in vals:
  with open("trans_%s.csv" % val) as f:
    c = csv.reader(f, delimiter=",")
    next(c) # Throw away the header
    #for _ in range(400):
    #  next(c) # Throw away bad data points
    vb = []
    ib = []
    ve = []
    for row in c:
      vb += [float(row[0])]
      ib += [float(row[1])]
      ve += [float(row[2])]

    vb_exp[val] = vb
    ib_exp[val] = ib
    ve_exp[val] = ve

valid = {}
valid[vals[0]] = (100, 960)
valid[vals[1]] = (200, 400)
valid[vals[2]] = (200, 300)
valid[vals[3]] = (200, 300)

params = curve_fit(lambda Vb, Av, b: Vb*Av+b, vb_exp[vals[0]][valid[vals[0]][0]:valid[vals[0]][1]], ve_exp[vals[0]][valid[vals[0]][0]:valid[vals[0]][1]])
Av, b = params[0][0], params[0][1]


fig = plt.figure()
ax = plt.subplot(111)
ax.plot(vb_exp[vals[0]], ve_exp[vals[0]], '.', label="Measured Emitter Voltage")
ax.plot(vb_exp[vals[0]], np.array(vb_exp[vals[0]])*Av+b, 'k-', label="Theorhetical Fit (Av = %f, b = %f)" % (Av,b))


for i in range(1,4):
  ax.plot(vb_exp[vals[i]],ve_exp[vals[i]], label="mR= " + vals[i][5:])
  params = curve_fit(lambda Vb, Av, b: Vb*Av+b, vb_exp[vals[i]][valid[vals[i]][0]:valid[vals[i]][1]], ve_exp[vals[i]][valid[vals[i]][0]:valid[vals[i]][1]])
  Av, b = params[0][0], params[0][1]
  ax.plot(vb_exp[vals[i]], np.array(vb_exp[vals[i]])*Av+b, 'k-', label="Theorhetical Fit (Av = %f, b = %f)" % (Av,b))

plt.title("Voltage Transfer Characteristic (VTC)")
plt.xlabel("Base Voltage (V)")
plt.ylabel("Emitter Voltage (V)")
plt.grid(True)
plt.ylim(0,6)
ax.legend()
plt.show()
# plt.savefig("consts.pdf")
