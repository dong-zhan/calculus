def func_CY3():		# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Spherical_harmonics_with_l_.3D_4

	global Y3_3, Y3_2, Y3_1, Y3, Y31, Y32, Y33, computeCoefficents, CEs, computeY
	
	def consts():
		global r, C_r_squared, C_r_cubed, CY3_3, CY3_2, CY3_1, CY3, CY31, CY32, CY33
		r = 1
		C_r_cubed = r * r * r
		C_r_squared = r * r
		CY3_3 	= 1./4. * np.sqrt( 35./		(2. * np.pi))
		CY3_2 	= 1./2. * np.sqrt( 105./	(1. * np.pi))
		CY3_1 	= 1./8. * np.sqrt( 21./		(2. * np.pi))
		CY3 	= 1./4. * np.sqrt( 7./		(1. * np.pi))
		CY31 	= 1./4. * np.sqrt( 21./		(2. * np.pi))
		CY32 	= 1./4. * np.sqrt( 105./	(1. * np.pi))
		CY33 	= -1./8. * np.sqrt( 35./		(2. * np.pi))
		
	consts()
	
	def Y3_1(x, y, z):
		return CY3_1 * (x - np.complex(0,1) * y) * (5 * z*z -  C_r_squared) / C_r_cubed	
	def Y3(x, y, z):
		return CY3 * (5*z*z - 3*C_r_squared) * z / C_r_cubed	
	def Y33(x, y, z):
		tmp = (x + np.complex(0,1)*y)
		return CY3 * tmp * tmp * tmp / C_r_cubed			

def vsh(m = 2, l = 3):				#Visualizing the spherical harmonics		(complex)
	import matplotlib.pyplot as plt
	from matplotlib import cm, colors
	from mpl_toolkits.mplot3d import Axes3D
	import numpy as np
	from scipy.special import sph_harm
	
	global x, y, z, sh, phi, theta, fcolors, fmin, fmax

	phi = np.linspace(0, np.pi, 100)
	theta = np.linspace(0, 2*np.pi, 100)
	phi, theta = np.meshgrid(phi, theta)

	x = np.sin(phi) * np.cos(theta)
	y = np.sin(phi) * np.sin(theta)
	z = np.cos(phi)

	sh = Y33(x, y, z)

	sh = np.sqrt(2) * (-1)**l * sh.real
	
	fmax, fmin = sh.max(), sh.min()
	
	fcolors = sh.real
	
	if fmax == fmin :
		pass
	else:
		fcolors = (fcolors - fmin)/(fmax - fmin)	
		
	mag = np.abs(sh)
	
	z = z * mag
	x = x * mag
	y = y * mag	
	
	fig = plt.figure(figsize=plt.figaspect(1.))
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
	#ax.set_axis_off()
	
	plt.show()	
	
	
	
	
