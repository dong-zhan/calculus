

def func(s):		# L{e^(-at)}
	global a
	return 1/(s+a)
	
def func(s):		
	global a
	return 1/(1+s*s)
	
def TransferFunction(s):		
	global a, b, z
	return (s*z+1) / ((s+a+np.complex(0,1)*b) * (s+a-np.complex(0,1)*b))
	
def plot_laplace(func, xstart, xstop, xstep, ystart, ystop, ystep, zmin, zmax) :
	global X,Y
	
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	
	#Z = np.zeros((2,2),dtype=np.complex_)
	
	X = np.arange(xstart, xstop, xstep)
	Y = np.arange(ystart, ystop, ystep)
	X, Y = np.meshgrid(X, Y)
	
	Z = np.vectorize(complex)(X, Y)
	
	Z = func(Z)

	# Plot the surface.
	surf = ax.plot_surface(X, Y, np.abs(Z), cmap=cm.coolwarm,
						   linewidth=0, antialiased=False)

	# Customize the z axis.
	ax.set_zlim(zmin, zmax)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()
	
a = np.complex(1.5,1.5)
b = np.complex(0.5,0.5)
z = np.complex(1,1)

a = -1
b = 1
z = 0
plot_laplace(TransferFunction, -2, 2, 0.01, -2, 2, 0.01, 0, 22)
	
a = 12
plot_laplace(func, -1, 1, 0.01, -1, 1, 0.01, 0, 0.2)
	
plot_laplace(func, -10, 10, 0.1, -10, 10, 0.1, 0, 6)
		
