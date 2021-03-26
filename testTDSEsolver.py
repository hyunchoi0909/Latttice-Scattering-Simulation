# This is a test of the Runge Kutta Method by animating a 1d gaussian wave packet interacting with a thin wall of potential energy
# Later must be expanded to a 2d wave packet and with multivariable Runge-Kutta methods
# Also gotta fix Laplacian calculation


import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation

xlim = (0,10)
xpts = 1000
x = np.linspace(xlim[0],xlim[1],xpts)

# Define some arbitrary, unitless gaussian wave packet, and plot just to make sure
mu = 1
x_0 = 5
sigma = 0.1
k = 20
psi = mu * np.exp(-((x-x_0)**2)/(2 * (sigma)**2)) * np.exp(k * 1j * x)
plt.plot(x, np.absolute(psi))
plt.show()

# Define a function f to solve via the Runge-Kutta Method of the 4th order
# dpsi/dt = f(psi(x)) = (laplacian(psi(x,t)) - V(x)psi(x,t)) * i
def f(psi):
	global x, V
	# Calcuate x spacing
	dx = np.abs(x[0] - x[1])

	##############################################################
	# CHANGE LAPLACIAN CALCULATION METHOD (Issue with end cases) #
	##############################################################

	laplacian = (np.roll(psi,1) + np.roll(psi,-1) - 2*(psi)) / (2*dx**2)

	return (laplacian - V * psi) * 1j

# Calculate psi(x, t + dt) from psi(x,t) by using the Runge-Kutta Method of the 4th order
# This function takes in the current wave function and evolves it a specific time interval
def timestep():
	global psi
	# Set time interval; this decision is very finnicky
	dt = 0.0001

	# Stages of fourth order Runge Kutta Method
	k1 = f(psi) 
	k2 = f(psi + (dt/2) * k1) 
	k3 = f(psi + (dt/2) * k2)
	k4 = f(psi + dt * k3)
	psi = psi + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
	return

# Really badly coded potential function; approximates a thin wall of certain potential
V = []
for i in x:
	if i > 5.5 and i < 5.75:
		V.append(500)
	else:
		V.append(0)
print(V)
V = np.array(V)

# Plotting and animation
fig = plt.figure()
ax = plt.axes(xlim=(0,10), ylim=(0,1))
line, = ax.plot([],[], lw=2)
plt.axvline(x=5.5, c='r')
plt.axvline(x=5.75, c='r')

def init():
	line.set_data([],[])
	return line,

def animate(i):
	global x, psi
	line.set_data(x,np.absolute(psi))
	timestep()
	return line,

anim = animation.FuncAnimation(fig,animate, init_func=init, frames=100, interval=20, blit=True)
plt.show()