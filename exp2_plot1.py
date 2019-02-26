import csv
import numpy as np
import matplotlib.pyplot as plt

V1k = []
Ib1k = []
Ie1k = []

V10k = []
Ib10k = []
Ie10k = []

V100k = []
Ib100k = []
Ie100k = []

with open('trans_exp2_1k.csv') as f:
    c = csv.reader(f, delimiter=",")
    next(c)
    
    for row in c:
        V1k += [float(row[0])]
        Ib1k += [float(row[1])]
        Ie1k += [-float(row[2])]

with open('trans_exp2_10k.csv') as f:
    c = csv.reader(f, delimiter=",")
    next(c)
    
    for row in c:
        V10k += [float(row[0])]
        Ib10k += [float(row[1])]
        Ie10k += [-float(row[2])]

with open('trans_exp2_100k.csv') as f:
    c = csv.reader(f, delimiter=",")
    next(c)
    
    for row in c:
        V100k += [float(row[0])]
        Ib100k += [float(row[1])]
        Ie100k += [-float(row[2])]
        
fig = plt.figure()
ax = plt.subplot(111)

Ic1k = np.array(Ie1k) - np.array(Ib1k)
Ic10k = np.array(Ie10k) - np.array(Ib10k)
Ic100k = np.array(Ie100k) - np.array(Ib100k)


ax.semilogy(V1k,Ic1k,'g', label='1k')
ax.semilogy(V10k,Ic10k,'b', label='10k')
ax.semilogy(V100k,Ic100k,'r', label='100k')
plt.xlabel('Base Voltage (V)')
plt.ylabel('Collector Current (A)')
ax.legend()
#ax.semilogx(x,y)
plt.show()
