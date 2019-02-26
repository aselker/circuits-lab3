
import smu

def linspace(initial, final, n = 100):
    if n>=2:
        increment = (float(final) - float(initial))/(n - 1)
        return [float(initial) + i*increment for i in range(n)]
    else:
        return []

s = smu.smu()
iin = linspace(-0.001, 0.001, 101)
f = open('idivider.csv', 'w')
f.write('"Iin","Iout"\n')

# s.set_autorange(1, 1)
# s.set_autorange(2, 1)

s.set_voltage(2, 0.)
for i in iin:
    s.set_current(1, i)
    s.autorange(1)
    s.autorange(2)
    f.write('{!s},{!s}\n'.format(i, s.get_current(2)))

s.set_current(1, 0.)
s.set_current(2, 0.)
f.close()
