#!/usr/bin/env python3
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

valid = 350

def ic_f(Vbe, Ut, Is):
  return Is * (np.exp(Vbe/Ut) - 1)

params = curve_fit(lambda Vbe, Ut, Is: np.log(ic_f(Vbe, Ut, Is)), vb_exp[:valid], np.log(ic_exp[:valid]))
Ut, Is = params[0][0], params[0][1]

fig = plt.figure()
ax = plt.subplot(111)
ax.semilogy(vb_exp, ib_exp, 'b.', label="Measured base current")
ax.semilogy(vb_exp, ie_exp, 'g.', label="Measured emitter current")
ax.semilogy(vb_exp, ic_exp, 'r.', label="Calculated collector current")
ax.semilogy(vb_exp, ic_f(vb_exp, params[0][0], params[0][1]), 'k-', label="Theoretical collectur current (Ut = %.4g, Is = %.4g)" % (Ut, Is))

ax.axvline(vb_exp[valid])

plt.title("Transistor base voltage and currents")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True)
ax.legend()
plt.show()
# plt.savefig("consts.pdf")
