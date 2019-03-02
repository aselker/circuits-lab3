#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from scipy.optimize import minimize

vb_exp = [[],[],[]]
ib_exp = [[],[],[]]
ie_exp = [[],[],[]]

rnames = ['1k','10k','100k']

for i in range(3):
    with open('trans_exp2_%s.csv' % rnames[i]) as f:
        c = csv.reader(f, delimiter=",")
        next(c)
        for _ in range(600): # Throw away bad data
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

def ic_f(Vbe): return Is * (np.exp(Vbe/Ut) - 1)

def ib_f(Vbe): return (Is/β) * (np.exp(Vbe/Ut) - 1)

def ie_f(Vbe): return ic_f(Vbe) + ib_f(Vbe)

ve_t = [[],[],[]]
ie_t = [[],[],[]]
ic_t = [[],[],[]]
for j in range(3):
  r = [1000, 10000, 100000][j]

  def ir_f(Ve):
    return Ve / r

  def err(Vb, Ve): # Error is 0 iff KCL holds
    i1 = ie_f(Vb - Ve)
    i2 = ir_f(Ve)
    return abs(i1 - i2)

  for vb in vb_exp[j]:
    ve_t[j] += [minimize(lambda Ve: err(vb, Ve), 0, method='Nelder-Mead').x[0]]

  ie_t[j] = [ie_f(vbe[0] - vbe[1]) for vbe in zip(vb_exp[j], ve_t[j])]
  ic_t[j] = [ic_f(vbe[0] - vbe[1]) for vbe in zip(vb_exp[j], ve_t[j])]

fig = plt.figure()
ax = plt.subplot(111)

# Joined semilog V-I plot
for i in range(3):
  ax.semilogy(vb_exp[i], ic_exp[i], ['r.', 'g.', 'b.'][i], label="Measured collector current (" + rnames[i] + " Ω)")
  ax.semilogy(vb_exp[i], ic_t[i], ['k--', 'k-.', 'k:'][i], label="Theoretical collector current (" + rnames[i] + "Ω)")

plt.title("Emitter-Degenerated Collector Current")
plt.xlabel("Base voltage (V)")
plt.ylabel("Collector current (A)")
plt.grid(True)
ax.legend()
plt.savefig("exp2_ic_all.pdf")
ax.cla()


# Separate linear V-I plots
for i in range(3):
  ax.plot(vb_exp[i], ic_exp[i], ['r.', 'g.', 'b.'][i], label="Measured collector current")
  ax.plot(vb_exp[i], ic_t[i], ['k--', 'k-.', 'k:'][i], label="Theoretical collector current")
  plt.title("Emitter-Degenerated Collector Current (" + rnames[i] + " Ω)")
  plt.xlabel("Base voltage (V)")
  plt.ylabel("Collector current (A)")
  plt.grid(True)
  ax.legend()
  plt.savefig("exp2_ic_" + rnames[i] + ".pdf")
  ax.cla()

