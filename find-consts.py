#!/usr/bin/env python3
import csv
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

vb_exp = []
ib_exp = []
ie_exp = []

def ie_f(V, Ut, Is):
  return Is*(np.exp(V/Ut) - 1)

with open("trans.csv") as f:
  c = csv.reader(f, delimiter=",")
  next(c) # Throw away the header
  for _ in range(400):
    next(c) # Throw away bad data points
  for row in c:
    vb_exp += [float(row[0])]
    ib_exp += [float(row[1])]
    ie_exp += [-float(row[2])]


params = curve_fit(lambda V, Ut, Is: np.log(ie_f(V, Ut, Is)), ie_exp, np.log(vb_exp))
Ut, Is = params[0][0], params[0][1]
print("Ut = %g , Is = %g" % (Ut, Is))

fig = plt.figure()
ax = plt.subplot(111)
ax.semilogy(vb_exp, ie_exp, 'b.', label="Measured emitter current")
ax.semilogy(vb_exp, ie_f(vb_exp, params[0][0], params[0][1]), 'r-', label="Theoretical fit (Ut = %.4g, Is = %.4g)" % (Ut, Is))

plt.title("Transistor base voltage and emitter current")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True)
ax.legend()
plt.show()
# plt.savefig("consts.pdf")
