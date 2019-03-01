import csv
import numpy as np
import matplotlib.pyplot as plt

res = ['1k','10k','100k']

V = {}
Ib = {}
Ie = {}
Ic = {}
Icmod = {}

Is = 3.343531794569888e-14
Ut = 0.028064113795844548

for val in res:
    with open('trans_exp2_%s.csv' % val) as f:
        c = csv.reader(f, delimiter=",")
        next(c)
        R = int(val[:-1])*1000
        print(R)
        Vtmp = []
        Ibtmp = []
        Ietmp = []
        for row in c:
            Vtmp += [float(row[0])]
            Ibtmp += [float(row[1])]
            Ietmp += [-float(row[2])]
        Ictmp = np.array(Ietmp) - np.array(Ibtmp)
        Icmodel = Is*np.exp((np.array(Vtmp)-R*np.array(Ietmp))/Ut)
        V[val] = Vtmp
        Ib[val] = Ibtmp
        Ie[val] = Ietmp
        Ic[val] = Ictmp
        Icmod[val] = Icmodel

        
fig1k = plt.figure()
i = 1

for val in res:
    ax = plt.subplot(3,1,i)

    ax.semilogy(V[val],Ic[val])
    ax.semilogy(V[val],Icmod[val])
    #ax.plot(V[val],Ie[val])
    i += 1
    plt.xlabel('Base Voltage (V)')
    plt.ylabel('Collector Current (A)')
    plt.title(val + " Resistor")

#ax.semilogx(x,y)
plt.show()
