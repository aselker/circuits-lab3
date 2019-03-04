#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.special import lambertw

vb_exp = [[],[],[]]
ib_exp = [[],[],[]]
ie_exp = [[],[],[]]

rnames = ['1k','10k','100k']

for i in range(3):
    with open('trans_exp2_%s.csv' % rnames[i]) as f:
        c = csv.reader(f, delimiter=",")
        next(c)
        for _ in range(670): # Throw away bad data
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

"""
# These should not be used again
Ut = np.nan
Is = np.nan
β = np.nan
r = np.nan
"""

Ut_c, Is_c, β_c, r_c = [], [], [], []
ic_c = [[],[],[]]
ib_c = [[],[],[]]
for j in range(3):

  def ic_w(Vb, Ut, Is, α, r):
    ic = -Is + (Ut * α * lambertw(np.exp((Is*r+Vb*α)/(Ut*α))*Is*r/(Ut*α)))/r
    # if (ic.imag != 0).any():
    #  print("Nonzero imaginary part with params ", Ut, Is, α, r)
    return ic.real

  def ic_f(Vb, Ut, Is, β, r): 
    return ic_w(Vb, Ut, Is, β/(1+β), r)

  """
  params = curve_fit(lambda Vb, Ut, Is, β, r: np.log(ic_f(Vb, Ut, Is, β, r)), vb_exp[j], np.log(ic_exp[j]), p0 = [0.028, 3.343e-14, 177, 10000], bounds = (1e-18, [1, 1, np.inf, np.inf]))[0]
  Ut_c += [params[0]]
  Is_c += [params[1]]
  β_c += [params[2]]
  r_c += [params[3]]
  """

  """
  params = curve_fit(lambda Vb, Ut, β, r: np.log(ic_f(Vb, Ut, Is, β, r)), vb_exp[j], np.log(ic_exp[j]), p0 = [0.028, 177, 10000], bounds = (1e-18, [1, np.inf, np.inf]))[0]
  Ut_c += [params[0]]
  Is_c += [Is]
  β_c += [params[1]]
  r_c += [params[2]]

  print(params)

  ic_c[j] = [ic_f(Vb, Ut_c[j], Is_c[j], β_c[j], r_c[j]) for Vb in vb_exp[j]]
  err = err_f(ic_c[j])
  print("log rms error = ", err)
  """

  def err_f(ic): return np.mean(np.power(np.log(ic) - np.log(ic_exp[j]), 2))

  res = minimize(lambda args: err_f([ic_f(Vb, args[0], args[1], args[2], args[3]) for Vb in vb_exp[j]]), x0 = [0.028, 3.343e-14, 177, 10000], method='Nelder-Mead') #, bounds=[(0.015, 1), (1e-18, 1), (1e-18, np.inf), (1e-18, np.inf)], method='TNC')
  # res = minimize(lambda args: err_f([ic_f(Vb, args[0], args[1], args[2], args[3]) for Vb in vb_exp[j]]), x0 = [0.028, 3.343e-14, 177, 10000], bounds=[(0.015, 1), (1e-18, 1), (1e-18, np.inf), (1e-18, np.inf)], method='TNC')
  print(res)
  params = res.x
  Ut_c += [params[0]]
  Is_c += [params[1]]
  β_c += [params[2]]
  r_c += [params[3]]

  ic_c[j] = [ic_f(Vb, Ut_c[j], Is_c[j], β_c[j], r_c[j]) for Vb in vb_exp[j]]
  err = err_f(ic_c[j])
  print("log ms error = ", err)

  ib_c[j] = [ic/β_c for ic in ic_c[j]]


# Find incremental resistances
rb_exp, rb_c, gm_exp, gm_c = [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]
for j in range(3):
  rb_exp[j] = np.diff(vb_exp[j]) / np.diff(ib_exp[j])
  rb_c[j] = np.diff(vb_exp[j]) / np.diff(ib_c[j])
  gm_exp[j] = np.diff(ic_exp[j]) / np.diff(vb_exp[j])
  gm_c[j] = np.diff(ic_c[j]) / np.diff(vb_exp[j])


fig = plt.figure()
ax = plt.subplot(111)

# Joined semilog V-I plot
for i in range(3):
  ax.semilogy(vb_exp[i], ic_exp[i], ['r.', 'g.', 'b.'][i], label="Measured collector current (" + rnames[i] + " Ω)")
  ax.plot(vb_exp[i], ic_c[i], ['k--', 'k-.', 'k-'][i], label="Theoretical collector current (" + rnames[i] + "Ω)")

plt.title("Emitter-Degenerated Collector Current")
plt.xlabel("Base voltage (V)")
plt.ylabel("Collector current (A)")
plt.grid(True)
ax.legend()
# plt.show()
plt.savefig("exp2_ic_all.pdf")
ax.cla()


# Separate linear V-I plots
for i in range(3):
  ax.plot(vb_exp[i], ic_exp[i], ['r.', 'g.', 'b.'][i], label="Measured collector current")
  ax.plot(vb_exp[i], ic_c[i], ['k--', 'k-.', 'k-'][i], label="Theoretical collector current")
  plt.title("Emitter-Degenerated Collector Current (" + rnames[i] + " Ω)")
  plt.xlabel("Base voltage (V)")
  plt.ylabel("Collector current (A)")
  plt.grid(True)
  ax.legend()
  plt.savefig("exp2_ic_" + rnames[i] + ".pdf")
  ax.cla()


# Joined log-log inc. res. plot
  ax.loglog(ib_exp[i], rb_exp[i], ['r.', 'g.', 'b.'][i], label="Measured incremental base resistance (" + rnames[i] + " Ω)")
  ax.plot(ib_exp[i], rb_c[i], ['k--', 'k-.', 'k-'][i], label="Theoretical incremental base resistance (" + rnames[i] + "Ω)")

plt.title("Emitter-Degenerated Incremental Base Resistance")
plt.xlabel("Base current (A)")
plt.ylabel("Incremental resistance (Ω)")
plt.grid(True)
ax.legend()
# plt.show()
plt.savefig("exp2_inc_res.pdf")
ax.cla()
