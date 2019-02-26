#!/usr/bin/env python2
import smu
import numpy as np

s = smu.smu()
v = np.arange(0., 1., 1./100.)
f = open("diode-2.csv", 'w')
f.write('"V","I"\n')

for val in v:
    s.set_voltage(1, val)
    s.autorange(1)
    f.write('{!s},{!s}\n'.format(val, s.get_current(1)))

s.set_voltage(1, 0.)
f.close()
