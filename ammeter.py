import smu
s = smu.smu()


i = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

try:
  while True:
    s.set_voltage(1, 2.0)
    s.autorange(1)
    s.set_voltage(2, 0)
    s.autorange(2)
    # i = [s.get_current(2)] + i[:-1]
    # print(sum(i) / len(i))
    print(s.get_current(2))
    pass
except KeyboardInterrupt:
  s.set_voltage(1, 0.)
  s.set_voltage(2, 0.)
