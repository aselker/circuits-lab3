import csv
import numpy as np
import matplotlib.pyplot as plt

res = ['1k','10k','100k']

V = {}
Ib = {}
Ie = {}
Ic = {}

for val in res:
    with open('trans_exp2_%s.csv' % val) as f:
        c = csv.reader(f, delimiter=",")
        next(c)
        for i in range(0,600):
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
        V[val] = Vtmp
        Ib[val] = Ibtmp
        Ie[val] = Ietmp
        Ic[val] = Ictmp

        
fig = plt.figure()
ax = plt.subplot(111)

for val in res:

    ax.semilogy(V[val],Ic[val], label=val)
    plt.xlabel('Base Voltage (V)')
    plt.ylabel('Collector Current (A)')
    #plt.title(val + " Resistor")

plt.title("Collector Current vs Base Voltage")
plt.legend()
#ax.semilogx(x,y)
plt.show()
