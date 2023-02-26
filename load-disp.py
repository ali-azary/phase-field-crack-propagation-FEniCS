#!/usr/bin/env python
from itertools import *
from matplotlib.pyplot import *
from numpy import *

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18}

fig, ax = subplots()

x=[0.]
y=[0.]
f=open('./results/output','r')
for line in f:
    try:
        x.append(float(real(line.split())[0]))
        y.append(float(real(line.split())[1]))
    except:
        pass

plot(x[:min(size(x),size(y))],y[:min(size(x),size(y))],'k',label='numerical solution')
xlabel('displacement')
ylabel('load')
savefig('load-disp.jpg')
