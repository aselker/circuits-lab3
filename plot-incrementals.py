#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt

V = []
Ib = []
Ie = []

with open("trans.csv") as f:
    c = csv.reader(f, delimiter=",")
    next(c)
    for i in range(0):  # TODO: Cut down the data properly
        next(c)
    for row in c:
        V += [float(row[0])]
        Ib += [float(row[1])]
        Ie += [-float(row[2])]

Ic = np.array(Ie) - np.array(Ib)

# TODO: Make the theoretial fits actually functions of the X-axis variable (base current)

# Constants from find-consts.py
Ut = 0.0280641
Is = 3.34353e-14
β = 177.098


def ib_f(Vbe):
    return (Is / β) * (np.exp(Vbe / Ut) - 1)


def ic_f(Vbe):
    return Is * (np.exp(Vbe / Ut) - 1)


rb_exp = np.diff(V) / np.diff(Ib)  # Incremental base resistance
rb_t = np.diff(V) / np.diff([ib_f(v) for v in V])

rb_v = np.arange(min(V), max(V), (max(V) - min(V)) / len(V))
rb_t_both = [[], []]
rb_t_both[0] = [ib_f(v) for v in rb_v][:-1]
rb_t_both[1] = np.diff(rb_v) / np.diff([ib_f(v) for v in rb_v])

gm_exp = np.diff(Ic) / np.diff(V)  # Incremental transconductance gain
gm_t = np.diff([ic_f(v) for v in V]) / np.diff(V)

gm_v = np.arange(min(V), max(V), (max(V) - min(V)) / len(V))
gm_t_both = [[], []]
gm_t_both[0] = [ic_f(v) for v in gm_v][:-1]
gm_t_both[1] = np.diff([ic_f(v) for v in gm_v]) / np.diff(gm_v)


def clip_range(xs, ys, bounds):
    pairs = [(x, y) for (x, y) in zip(xs, ys) if (bounds[0] <= y) and (y <= bounds[1])]
    return list(zip(*pairs))


rb_exp_plot = clip_range(Ib[:-1], rb_exp, (1e0, 1e11))
rb_t_plot = clip_range(rb_t_both[0], rb_t_both[1], (1e2, 1e7))

gm_exp_plot = clip_range(Ic[:-1], gm_exp, (1e-8, 1e1))
gm_t_plot = clip_range(gm_t_both[0], gm_t_both[1], (1e-7, 1e0))
# gm_exp_plot = clip_range(Ic[:-1], gm_exp, (-np.inf, np.inf))
# gm_t_plot = clip_range(gm_t_both[0], gm_t_both[1], (-np.inf, np.inf))


fig = plt.figure()
ax = plt.subplot(111)

ax.loglog(Ib[:-1], rb_exp, "b.", label="Measured Base Incremental Resistance")
ax.loglog(
    rb_t_plot[0], rb_t_plot[1], "g-", label="Theoretical Base Incremental Resistance"
)
plt.xlabel("Base current (A)")
plt.ylabel("Incremental Base Resistance (Ω)")
plt.title("Incremental Base Resistance")
ax.legend()
# plt.show()
plt.savefig("rb.pdf")
ax.cla()


ax.loglog(gm_exp_plot[0], gm_exp_plot[1], "b.", label="Measured Transconductance")
ax.loglog(gm_t_plot[0], gm_t_plot[1], "g-", label="Theoretical Transconductance")
plt.xlabel("Collector current (A)")
plt.ylabel("Incremental Transconductance (1/Ω)")
plt.title("Incremental Base-Collector Transconductance Gain")
ax.legend()
# plt.show()
plt.savefig("gm.pdf")
ax.cla()
