#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

vb_exp = []
ib_exp = []
ve_exp = []

with open("trans_exp3_10k.csv") as f:
  c = csv.reader(f, delimiter=",")
  next(c) # Throw away the header
  #for _ in range(400):
  #  next(c) # Throw away bad data points
  for row in c:
    vb_exp += [float(row[0])]
    ib_exp += [float(row[1])]
    ve_exp += [float(row[2])]

valid = (100, 960)

params = curve_fit(lambda Vb, Av, b: Vb*Av+b, vb_exp[valid[0]:valid[1]], ve_exp[valid[0]:valid[1]])
Av, b = params[0][0], params[0][1]


fig = plt.figure()
ax = plt.subplot(111)
ax.plot(vb_exp, ve_exp, '.', label="Measured Emitter Voltage")
ax.plot(vb_exp, np.array(vb_exp)*Av+b, 'k-', label="Theoretical Fit (Aᵥ = %f, V₀ = %f)" % (Av,b))

ax.axvline(vb_exp[valid[0]], label="Sampled Data")
ax.axvline(vb_exp[valid[1]])

plt.title("Voltage Transfer Characteristic (VTC)")
plt.xlabel("Base Voltage (V)")
plt.ylabel("Emitter Voltage (V)")
plt.grid(True)
ax.legend()
#plt.show()
plt.savefig("exp3_theoretical.pdf")
