#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit

vb_exp = [[],[],[]]
ib_exp = [[],[],[]]
ie_exp = [[],[],[]]

rnames = ['1k','10k','100k']

for i in range(3):
    with open('trans_exp2_%s.csv' % rnames[i]) as f:
        c = csv.reader(f, delimiter=",")
        next(c)
        for _ in range(800): # Throw away bad data - 600 for final, 800 for easier fit
          next(c)
        for row in c:
          vb_exp[i] += [float(row[0])]
          ib_exp[i] += [float(row[1])]
          ie_exp[i] += [-float(row[2])]


ic_exp = np.array(ie_exp) - np.array(ib_exp)


ve_t = [[],[],[]]
ie_t = [[],[],[]]
ic_t = [[],[],[]]
for j in range(3):

  r = [1000, 10000, 100000][j]

  # Values from experiment 1
  Ut = 0.0280641
  Is = 3.34353e-14
  β = 177.098

  def ic_f(Vbe): return Is * (np.exp(Vbe/Ut) - 1)
  def ib_f(Vbe): return (Is/β) * (np.exp(Vbe/Ut) - 1)
  def ie_f(Vbe): return ic_f(Vbe) + ib_f(Vbe)

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


for j in range(3):
  r = [1000, 10000, 100000][j]
  def ir_f(Ve):
    return Ve / r

  # Do it again, but find the constants this time
  def err_f(Vb, Ut, Is, β):
    def ic_f(Vbe): return Is * (np.exp(Vbe/Ut) - 1)
    def ib_f(Vbe): return (Is/β) * (np.exp(Vbe/Ut) - 1)
    def ie_f(Vbe): return ic_f(Vbe) + ib_f(Vbe)
    return lambda Ve: abs(ie_f(Vb - Ve) - ir_f(Ve))

  def ic_f(Vb, Ut, Is, β): 
    # print(Vb, Ut, Is, β)
    err = err_f(Vb, Ut, Is, β)
    Ve = minimize(err, 0,  method='Nelder-Mead').x[0]
    return Is*(np.exp((Vb - Ve)/Ut) - 1)

  params = curve_fit(lambda Vb, Ut, Is, β: [np.log(ic_f(this_Vb, Ut, Is, β)) for this_Vb in Vb], vb_exp[j], np.log(ic_exp[j]))
  print(params)
  # curve_fit(ic_f, vb_exp[j], ic_exp[j])

  # def f_temp(x, a, b): return b * ( np.exp(x/a) - 1)

  #def f_temp(x, n): return ic_f(x, n, 1, 1)
  #print(curve_fit(lambda x, n: [f_temp(xx, n) for xx in x], vb_exp[j], ic_exp[j]))
  # print(ic_f(0,0,1,1))


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

