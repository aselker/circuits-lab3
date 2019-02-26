import csv
import numpy as np
import matplotlib.pyplot as plt

V = []
Ib = []
Ie = []

with open('trans.csv') as f:
    c = csv.reader(f, delimiter=",")
    next(c)
    for i in range(0,400):
        next(c)
    
    for row in c:
        V += [float(row[0])]
        Ib += [float(row[1])]
        Ie += [-float(row[2])]
        
fig = plt.figure()
ax = plt.subplot(111)

Ic = np.array(Ie) - np.array(Ib)

ax.semilogy(V,Ib,'r', label='Ib')
#ax.semilogy(V,Ie,'b')
ax.semilogy(V,Ic,'g', label='Ic')
plt.xlabel('Vb (V)')
plt.ylabel('Current (A)')
ax.legend()
#ax.semilogx(x,y)
plt.show()
