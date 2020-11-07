import numpy as np
import matplotlib.pyplot as plt


t= np.linspace(0,5,1001)
x1=2 * np.cos(10*t)
fig, ax = plt.subplots(figsize=(6,2.5))
ax.plot(t,x1)
ax.set_xlim(0,5)
ax.set_xlabel('t(sec)')
ax.set_title('$x_1(t)=2\cos(10t)$')
fig.show()


t= np.arange(-1,3,0.001)
x2=np.exp(-t) * (t>0)
fig, ax =plt.subplots(figsize=(6,2.5))
ax.plot(t, x2)
ax.set_xlim(-1,3)
ax.set_xlabel('t(sec)')
ax.set_title('$x_2(t)=e^{-t}u(t)$')
fig.show()


t= np.arange(-1,3,0.001)
x1= np.sin(np.pi * t) *((0<=t)&(t<2))
x2= 1.0 *((0<=t)&(t<1)) -1.0*((1<=t)&(t<2))

g1 = np.abs(x1)
g2 = x2**2
g3 = x1 + x2
g4 = x1 * x2

fig, axes = plt.subplots(3,2, figsize =(6,6.5))
axes[0,0].plot(t,x1)
axes[0,1].plot(t,x2)
axes[1,0].plot(t,g1)
axes[1,1].plot(t,g2)
axes[2,0].plot(t,g3)
axes[2,1].plot(t,g4)


#def x(t):
#    return t*((0<=t)&(t<1)) +2.0*((1<=t)&(t<2))

x = lambda t: t*((0<=t)&(t<1)) +2.0 * ((1<=t)&(t<2))
t= np.arange(-1,5,0.001)
x1 =lambda t: x(t-1)
x2 =lambda t: x1(2*t)
x3 =lambda t: x(2*t)
x4 =lambda t: x3(t-0.5)

fig, axes = plt.subplots(3,2, figsize =(6,6.5))
fig.delaxes(axes[0,1])

axes[0,0].plot(t,x(t))

axes[1,0].plot(t,x1(t))
axes[1,1].plot(t,x2(t))
axes[2,0].plot(t,x3(t))
axes[2,1].plot(t,x4(t))



x = lambda t: 1.0*((0<=t)&(t<1)) -0.5 * ((1<=t)&(t<3))
t= np.arange(-4, 4, 0.001)
xeven =lambda t: (x(t) +x(-t))/2
xodd =lambda t: (x(t) - x(-t))/2

fig, axes = plt.subplots(2,2, figsize =(6,4.5))

fig.delaxes(axes[0,1])

axes[0,0].plot(t,x(t))

axes[1,0].plot(t,xeven(t))
axes[1,1].plot(t,xodd(t))


square = lambda t : (abs(t%1)<0.5).astype(float)
t= np.arange(-2, 2, 0.001)
sawtooth = lambda t : (t%1)
triangular = lambda t : 2.0* (t%1) *(abs(t%1)<0.5) + 2.0*(1-1*abs(t%1))*(abs(t%1)>0.5) 
fwr =lambda t : np.sin(np.pi *(t%1))
fig, axes = plt.subplots(2,2, figsize =(6,4.5))

axes[0,0].plot(t,square(t))
axes[0,1].plot(t,sawtooth(t))
axes[1,0].plot(t,triangular(t))
axes[1,1].plot(t,fwr(t))


from numpy import random
t=np.linspace(0,1,1001)
signal =np.sin(2*np.pi*2*t)
noise = 0.5* (random.rand(len(t)) -0.5)
noisy_signal =signal +noise
fig, ax = plt.subplots(figsize =(6,2.5))

ax.plot(t, noisy_signal)

from sympy import *

x, t, A = symbols('x t A')
x = A * cos(2*pi* t)
integrate(x**2, (t,0,1))


#########################

#y = lambda t: x(t)**2

def Sys(x):
    return lambda t: x(t)**2

x1 = lambda t: np.exp(-2 * t) * np.sin(10 * np.pi * t)
x2 = lambda t: 0.5 * np.sin(2 * np.pi * t)*(t>0)
t = np.linspace(0, 1, 1001)
y1 =Sys(x1)(t)
y2 = Sys(x2)(t)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6, 2.55))
ax1.plot(t, x1(t), 'C0', lw = 0.5, label = 'x1') 
ax1.plot(t, x2(t), 'C2', lw = 1.5, label = 'x2')
ax1.set_xlim(0,1)
ax1.set_xlabel('t(sec)')
ax1.set_title('x1 andf x2')
ax1.legend(loc =1, fontsize=10)


ax2.plot(t, y1, 'C0', lw = 0.5, label = 'x1') 
ax2.plot(t, y2, 'C2', lw = 1.5, label = 'x2')
ax2.set_xlim(0,1)
ax2.set_xlabel('t(sec)')
ax2.set_title('y1 andf y2')
ax2.legend(loc =1, fontsize=10)


########################
def Sys(x):
    return lambda t: (x(t))**3

t = np.linspace(-1, 3, 1001)
x1 = lambda t: 1.0 * ((0<= t) &(t <1)) -1 * ((1<= t) &(t <2))
x2 = lambda t:  np.cos(np.pi * t)*((0<= t) &(t <2))

x12 = lambda t: x1(t) +x2(t)

w=Sys(x12)(t)
y=Sys(x1)(t) + Sys(x2)(t)

fig, axes = plt.subplots(2, 2, figsize = (6, 4.5))
axes[0,0].plot(t,x1(t))
axes[0,1].plot(t,x2(t))
axes[1,0].plot(t,w)
axes[1,1].plot(t,y)


###################
def Sys(x):
    return lambda t: np.cos(np.pi * t) * x(t)

t = np.arange(-2, 6, 0.001)
x = lambda t: 1.0 * ((0<= t) &(t <2)) -1 * ((2<= t) &(t <4))
t0 = 1
v=Sys(x )(t)
w=Sys(lambda t: x(t - t0) )(t)
y = Sys(x)(t -t0)


fig, axes = plt.subplots(2, 2, figsize = (6, 4.5))
axes[0,0].plot(t,x(t))
axes[0,1].plot(t,v)
axes[1,0].plot(t,w)
axes[1,1].plot(t,y)


############################
from scipy import integrate

x = lambda t: t * ((0<= t) &(t <1)) +(2-t) * ((1<= t) &(t < 2))
h = lambda t: 1.0 * ((0<= t) &(t <1)) -1 * ((1<= t) &(t < 2))


t = np.linspace(-1, 5, 1001)
tau = np.linspace(-1, 5, 1001)

y= np.zeros(len(t))


for n, t0 in enumerate(t):
    prod = lambda tau : x(tau) * h(t0 -tau)
    y[n] =integrate.simps(prod(tau), tau)

fig, axes = plt.subplots(2, 2, figsize = (6, 4.5))
axes[0,0].plot(t,x(t))
axes[0,1].plot(t,h(t))
axes[1,0].plot(t,y)
axes[1,1].plot(t,y)

############################
from scipy import integrate

x = lambda t: 2 * ((0<= t) &(t <10)) 
h1 = lambda t: 2* np.exp(-0.2 *t) * (0<= t) 
h2 = lambda t: 2* np.exp(-1 *t) * (0<= t) 

t = np.linspace(-1, 20, 1001)
tau = np.linspace(-1,20, 1001)

y1= np.zeros(len(t))
y2= np.zeros(len(t))

for n, t0 in enumerate(t):
    prod1 = lambda tau : x(tau) * h1(t0 -tau)
    y1[n] =integrate.simps(prod1(tau), tau)
    prod2 = lambda tau : x(tau) * h2(t0 -tau)
    y2[n] =integrate.simps(prod2(tau), tau)


fig, axes = plt.subplots(3, 2, figsize = (18, 7))
axes[0,0].plot(t,x(t))
axes[0,1].plot(t,x(t))
axes[1,0].plot(t,h1(t))
axes[1,1].plot(t,h2(t))
axes[2,0].plot(t,y1)
axes[2,1].plot(t,y2)


####################################

x = lambda t: t * ((0<= t) &(t <1)) +(2-t) * ((1<= t) &(t < 2))
h = lambda t: 1.0 * ((0<= t) &(t <1)) -1 * ((1<= t) &(t < 2))

Ts =0.001
t = np.arange(-1, 5, Ts)
y= np.convolve(x(t), h(t))*Ts
ty = np.arange(-2, 10, Ts)[:-1]

fig, axes = plt.subplots(2, 2, figsize = (6, 4.5))
axes[0,0].plot(t,x(t))
axes[0,1].plot(t,h(t))
axes[1,0].plot(ty,y)
axes[1,1].plot(ty,y)

##################sympy를 이용한 미분방정식 해#########################################
from sympy import *
t, x = symbols("t, x", real =True, positive = True)
y = symbols("y", cls=Function)
print(y(t))

x = exp(-3*t)
def ode(y, homogenous =True):
    h =diff(y,t ) + 2*y
    return h if homogenous else h - 2*x

yh =dsolve(ode(y(t)))
print(yh)

C2 =symbols("C2")
yp = C2*exp(-3* t)
eqs =ode(yp,homogenous =False)
const = solve(eqs, C2)

yp = simplify(yp.subs(C2, const[0]))

print(yp)

yc =yh.rhs + yp
print(yc)

C1 = symbols("C1")
const =solve(yc.subs(t,0))
print(const)

yc = yc.subs(C1, const[0])
print(yc)

lam_yc = lambdify(t, yc)
lam_t =np.linspace(0, 5, 1001)
fig, ax = plt.subplots(figsize=(6,2.5))
ax.plot(lam_t, lam_yc(lam_t))


##############일반적인 미분방정식##
from sympy import *
          
x, y, z, t = symbols('x y z t')
f, g, h = symbols('f, g, h', cls=Function)

init_printing()

y = symbols('y', cls=Function)
y(t)

deq = Eq( y(t).diff(t), -2*y(t) )
deq

psol1= dsolve( deq, y(t) ) 
psol1
psol1 = psol1.rhs +0.5
psol1

C1 = symbols("C1")

eqs =[psol1.subs(t,0)]
const=solve(eqs,C1)
const

psol1=psol1.subs(C1, const[C1])
psol1

psol = dsolve( deq, y(t),ics= {y(0):0.5} )
psol

plot( psol1, xlim=(0,2), ylim=(0,1) )


lam_yc = lambdify(t, psol1)
lam_t =np.linspace(0, 20, 1001)
fig, ax = plt.subplots(figsize=(6,2.5))
ax.plot(lam_t, lam_yc(lam_t))


y(x)
deq = Eq( y(x).diff(x) + y(x)*tan(x), sin(2*x) )
deq

psol2= dsolve( deq, y(x) )
psol2

psol3 = dsolve( deq, ics= {y(0):1} )
psol3

y(t)
deq = Eq( 10*y(t).diff(t,2) + 10*y(t).diff(t) + 90*y(t), 0 )
deq

sol = dsolve( deq, ics= { y(0):0.16, y(t).diff(t).subs(t,0):0 } )
sol

y1, y2 = symbols('y1 y2', cls=Function)
y1(t)
y2(t)

eq1 = Eq( y1(t).diff(t), -0.02*y1(t)+0.02*y2(t) )
eq1

eq2 = Eq( y2(t).diff(t), 0.02*y1(t)-0.02*y2(t) )
eq2

dsolve( [ eq1, eq2 ] )

sol = dsolve( [ eq1, eq2 ], ics= { y1(0):150, y2(0):0 } )




###############################################
from sympy import *
t = symbols("t", positive =True, real =True)
y = symbols("y", cls=Function)
a1,a0 =1,1
x = 1
def ode(y, homogeneous =True):
    h =diff(y, t, t) + a1 *diff(y, t) + a0*y
    return h if homogeneous else h - x
ode(y(t))
yh = dsolve(ode(y(t)))
yh
C3 = symbols("C3")
yp = C3
eps = ode(yp, homogeneous = False)
eps
const =solve(eps, C3)
print(const)
yp=simplify(yp.subs(C3, const[0]))
yp
yc = yh.rhs + yp
yc
C1, C2 =symbols("C1, C2")
eqs =[yc.subs(t,0), yc.diff(t).subs(t,0)]
const=solve(eqs,[C1,C2])
const
yc = yc.subs(C1,const[C1]).subs(C2,const[C2])
print(simplify(yc))

lam_yc = lambdify(t, yc)
lam_t =np.linspace(0, 20, 1001)
fig, ax = plt.subplots(figsize=(6,2.5))
ax.plot(lam_t, lam_yc(lam_t))

###################################################
from scipy.integrate import odeint

def firstDE(y, t):
    return 2*np.exp(-3*t) - 2*y
    
yinit =0
t=np.linspace(0, 5,1001)
yc = odeint(firstDE, yinit, t)


t_anal =np.linspace(0,5,11)
y_anal = 2* np.exp(-2*t_anal) - 2*np.exp(-3*t_anal)

fig, ax = plt.subplots(figsize =(6,2.5))
ax.plot(t, yc, label='using ODE solve')
ax.plot(t_anal, y_anal, 'o', label = 'analytic solution')



#########################################################

from scipy import signal

t =np.linspace(0,16,501)
x=1.0*(t>0)
b= 1
a =[1,1,1]
x0=[0,0]

t,y,x_state =signal.lsim((b,a), x, t, x0)


#t_anal =np.linspace(0,16,16)
#y_anal = 2* np.exp(-2*t_anal) - 2*np.exp(-3*t_anal)

fig, ax = plt.subplots(figsize =(6,2.5))
ax.plot(t, y, label='using ODE solve')
#ax.plot(t_anal, y_anal, 'o', label = 'analytic solution')


#########################################################

from scipy import signal
b=1
a=[1,1,1]

t=np.linspace(0,5,101)
t, yimp = signal.impulse((b,a), T=t)
t, ystep =signal.step((b,a), T=t)

fig, ax = plt.subplots(figsize=(6,2.5))
ax.plot(t, yimp)
ax.plot(t, ystep, lw=2, color='0.7')



########################################



t= np.arange(-3,3,0.001)
x1= np.sin(np.pi * t) 
x2= np.sin(np.pi * t+1) 

g1 = x1 + x2
g2 = x1 - x2
g3 = x1 * x2
g4 = x1 * x2

fig, axes = plt.subplots(3,2, figsize =(6,6.5))
axes[0,0].plot(t,x1)
axes[0,1].plot(t,x2)
axes[1,0].plot(t,g1)
axes[1,1].plot(t,g2)
axes[2,0].plot(t,g3)
axes[2,1].plot(t,g4)


######################################
t= np.arange(-20,20,0.001)
x1= np.sin(np.pi * t) 
x2= np.sin(np.pi * t+1) 
x3= np.sin((np.pi+0.2) * t)

g1 = x1 + x3
g2 = x1 - x3
g3 = x1 * x3
g4 = x1 + x3


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 6))
ax1.plot(t, x1, 'C0', lw = 0.5, label = 'x1') 
ax1.plot(t, x3, 'C2', lw = 0.5, label = 'x2')
ax1.plot(t, g1, 'C3', lw = 1.5, label = 'x1 + x2')
ax1.plot(t, g2, 'C4', lw = 1.5, label = 'x1 + x2')
ax1.set_xlim(-20,20)
ax1.set_xlabel('t(sec)')
ax1.set_title('x1 andf x2')
ax1.legend(loc =1, fontsize=10)

ax2.plot(t, x1, 'C0', lw = 0.5, label = 'x1') 
ax2.plot(t, x3, 'C2', lw = 0.5, label = 'x2')
ax2.plot(t, g3, 'C3', lw = 1.5, label = 'x1 + x2')
ax2.plot(t, g4, 'C4', lw = 1.5, label = 'x1 + x2')
ax2.set_xlim(-20,20)
ax2.set_xlabel('t(sec)')
ax2.set_title('x1 andf x2')
ax2.legend(loc =1, fontsize=10)

############################################
def beat(A, B, fc, df, dur, fs=11025):
    """
    inputs:
        A= 



    """
    t= np.arange(0,dur, 1/fs)
    x1 = A * np.cos(2*np.pi*(fc-df)*t)
    x2 = A * np.cos(2*np.pi*(fc+df)*t)
    x= x1+ x2
    return x, t

from scipy import signal
fc, delf, fs, dur = 440, 5, 11025,3
x,t=beat(1, 0.5, fc, delf, dur)
fig, ax =plt.subplots(figsize = (15, 6))
ax.plot(t*1000, x)



from IPython.display import Audio
Audio(data =x, rate=fs)
























