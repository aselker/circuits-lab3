#!/usr/bin/env python2
import smu
import numpy as np

s = smu.smu()
v = np.logspace(-9, -2, 50)
f = open("diode-1.csv", 'w')
f.write('"I","V"\n')

for val in v:
    s.set_current(1, val)
    s.autorange(1)
    f.write('{!s},{!s}\n'.format(val, s.get_voltage(1)))

s.set_current(1, 0.)
f.close()
