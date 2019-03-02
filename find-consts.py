#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

vb_exp = []
ib_exp = []
ie_exp = []

with open("trans.csv") as f:
  c = csv.reader(f, delimiter=",")
  next(c) # Throw away the header
  for _ in range(400):
    next(c) # Throw away bad data points
  for row in c:
    vb_exp += [float(row[0])]
    ib_exp += [float(row[1])]
    ie_exp += [-float(row[2])]


ic_exp = np.array(ie_exp) - np.array(ib_exp)

valid = (100, 350)

def ic_f(Vbe, Ut, Is):
  return Is * (np.exp(Vbe/Ut) - 1)

params = curve_fit(lambda Vbe, Ut, Is: np.log(ic_f(Vbe, Ut, Is)), vb_exp[:valid[1]], np.log(ic_exp[:valid[1]]))
Ut, Is = params[0][0], params[0][1]

def ib_f(Vbe, β):
  return (Is/β) * (np.exp(Vbe/Ut) - 1)

params = curve_fit(lambda Vbe, β: np.log(ib_f(Vbe, β)), vb_exp[valid[0]:valid[1]], np.log(ib_exp[valid[0]:valid[1]]))
β = params[0][0]
print("Ut = %g, Is = %g, β = %g" % (Ut, Is, β))

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
plt.xlabel("Base voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True)
ax.legend()
plt.show()
# plt.savefig("consts.pdf")
