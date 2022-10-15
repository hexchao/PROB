import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import random



x0 = 0
x1 = 0
alpha = 0.01
tau_list = np.linspace(0,1,50)
dist2 = [[] for j in range(len(tau_list))]
dist2_list = []
for j in range((len(tau_list))):
    x0 = 0
    x1 = 0
    tau = tau_list[j]
    for i in range(1000):
        if x1-x0>=tau:
            g1=1
        elif x1-x0<=-tau:
            g1=-1
        elif abs(x1-x0)<=tau:
            g1=x1-x0
        x1=x1-alpha*(x1-1+g1)

        if x0-x1>=tau:
            g0=1
        elif x1-x0<=-tau:
            g0=-1
        elif abs(x0-x1)<=tau:
            g0=x0-x1
        x0=x0-alpha*(x0+g0)

        dist_temp = (x0-1/3)**2 + (x1-2/3)**2
        dist2[j].append(dist_temp)

for k in range((len(tau_list))):
    dist2_list.append(dist2[k][999])



x0 = 0
x1 = 0
alpha = 0.01
tau_list = np.linspace(0,1,50)
dist3 = [[] for j in range(len(tau_list))]
for j in range((len(tau_list))):
    tau = tau_list[j]
    x0 = 0
    x1 = 0
    for i in range(1000):
        g1=x1-x0
        x1=x1-alpha*(x1-1+g1)
        if x0-x1>=tau:
            g0=1
        elif x1-x0<=-tau:
            g0=-1
        elif abs(x0-x1)<=tau:
            g0=x0-x1
        x0=x0-alpha*(x0+g0)

        dist_temp = (x0-1/3)**2 + (x1-2/3)**2
        dist3[j].append(dist_temp)

dist3_list = []
for j in range((len(tau_list))):
    dist3_list.append(dist3[j][999])

del dist2_list[0]
del dist3_list[0]



pd.DataFrame(dist2).to_csv('F:/Simulation_CIFAR10/Censor HNRSA/16.0/LargeHuber/HuberCase/Dist2.csv')
pd.DataFrame(dist3).to_csv('F:/Simulation_CIFAR10/Censor HNRSA/16.0/LargeHuber/HuberCase/Dist3.csv')













'''
======================================================
'''


plt.plot(range(len(dist2_list)),dist2_list, '-o', markersize=12, label='Huber+Huber',linewidth=4)
plt.plot(range(len(dist3_list)),dist3_list, '-^', markersize=12, label='Huber+L2',linewidth=4)
plt.grid()
plt.xticks(range(0,60,10), [0,0.2,0.4,0.6,0.8,1.0])
plt.ylim(0, 0.5)
#plt.xlabel(r'$\tau$', fontsize=22)
plt.xlabel(r'$T_h$', fontsize=22)
plt.ylabel('Squared Error', fontsize=22)
plt.legend(loc='lower right',fontsize=22)
#plt.title('Comparison', fontsize=22)
plt.tick_params(labelsize=22)
plt.tight_layout()
plt.savefig('F:/Simulation_CIFAR10/Censor HNRSA/16.0/LargeHuber/HuberCase/Huber.png')
plt.close()#plt.show()



