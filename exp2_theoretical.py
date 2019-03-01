#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

vb_exp = [[],[],[]]
ib_exp = [[],[],[]]
ie_exp = [[],[],[]]

res = ['1k','10k','100k']

for i in range(3):
    with open('trans_exp2_%s.csv' % res[i]) as f:
        c = csv.reader(f, delimiter=",")
        next(c)
        for _ in range(0): # Throw away bad data
          next(c)
        for row in c:
          vb_exp[i] += [float(row[0])]
          ib_exp[i] += [float(row[1])]
          ie_exp[i] += [-float(row[2])]


ic_exp = np.array(ie_exp) - np.array(ib_exp)

valid = (100, 350)

# Values from find-consts.py
Ut = 0.0280641
Is = 3.34353e-14
β = 177.098

def ic_f(Vbe):
  return Is * (np.exp(Vbe/Ut) - 1)

def ib_f(Vbe):
  return (Is/β) * (np.exp(Vbe/Ut) - 1)

ve_t = [[],[],[]]
for j in range(3):
  r = [1000, 10000, 100000][j]

  def ir_f(Ve):
    return Ve / r

  def err(Vb, Ve): # Error is 0 iff KCL holds
    i1 = ic_f(Vb - Ve)
    i2 = ir_f(Ve)
    return abs(i1 - i2)

  for vb in vb_exp[j]:
    ve_t += [minimize(lambda Ve: err(vb, Ve), 0, method='Nelder-Mead').x[0]]

"""

fig = plt.figure()
ax = plt.subplot(111)
ax.semilogy(vb_exp, ib_exp, 'b.', label="Measured base current")
ax.semilogy(vb_exp, ie_exp, 'g.', label="Measured emitter current")
ax.semilogy(vb_exp, ic_exp, 'r.', label="Calculated collector current")
ax.semilogy(vb_exp, ic_f(vb_exp, Ut, Is), 'k-', label="Theoretical collector current (Ut = %.4g, Is = %.4g)" % (Ut, Is))
ax.semilogy(vb_exp, ib_f(vb_exp, β), 'k--', label="Theoretical base current (Ut = %.4g, Is = %.4g, β = %.4g)" % (Ut, Is, β))

ax.axvline(vb_exp[valid[0]])
ax.axvline(vb_exp[valid[1]])

plt.title("Transistor base voltage and currents")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True)
ax.legend()
plt.show()
# plt.savefig("consts.pdf")
"""
