def pf(df, xstart, xstop, xnum = 50, show = True, linewidth=1.0):		
	global plt, t, s, xstep
	
	xstep = (xstop-xstart)/(xnum)
	
	t = np.arange(xstart, xstop+xstep, xstep)			#use xstop+xstep to include the endpoint.
	s = df(t)
	line, = plt.plot(t, s, linewidth = linewidth)		
	
	if show:
		plt.show()
		

#########################################################

# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Spherical_harmonics_with_l_.3D_4
def func_Y():			# real
	global Y0, Y1_1, Y1, Y11, Y2_2, Y2_1, Y2, Y21, Y22, Y3_3, Y3_2, Y3_1, Y3, Y31, Y32, Y33, computeCoefficents, CEs, computeY, getY, Y0Table, Y1Table, Y2Table, Y3Table, YTable, computeCoefficentsY3, computeCoefficentsY2, computeCoefficentsY1, computeCoefficentsY0, computeY3, computeY2, computeY1, computeY0
	
	def consts():		# real
		global r, C_r_squared, C_r_cubed, CY0, CY1_1, CY1, CY11, CY2_2, CY2_1, CY2, CY21, CY22, CY3_3, CY3_2, CY3_1, CY3, CY31, CY32, CY33
			
		r = 1
		C_r_cubed = r * r * r
		C_r_squared = r * r
		
		CY0		= 1./2. * np.sqrt(1./		(1. * np.pi))
		
		CY1_1	= 1./1. * np.sqrt(3./		(4. * np.pi))		#l,m
		CY1		= 1./1. * np.sqrt(3./		(4. * np.pi))
		CY11	= 1./1. * np.sqrt(3./		(4. * np.pi))
		
		CY2_2	= 1./2. * np.sqrt(15./		(1. * np.pi))
		CY2_1	= 1./2. * np.sqrt(15./		(1. * np.pi))
		CY2		= 1./4. * np.sqrt(5./		(1. * np.pi))
		CY21	= 1./2. * np.sqrt(15./		(1. * np.pi))		
		CY22	= 1./4. * np.sqrt(15./		(1. * np.pi))		
		
		CY3_3 	= 1./4. * np.sqrt( 35./		(2. * np.pi))
		CY3_2 	= 1./2. * np.sqrt( 105./	(1. * np.pi))
		CY3_1 	= 1./4. * np.sqrt( 21./		(2. * np.pi))
		CY3 	= 1./4. * np.sqrt( 7./		(1. * np.pi))
		CY31 	= 1./4. * np.sqrt( 21./		(2. * np.pi))
		CY32 	= 1./4. * np.sqrt( 105./	(1. * np.pi))
		CY33 	= 1./4. * np.sqrt( 35./		(2. * np.pi))
		
	consts()
	
	################ Y0 ################
	def Y0(x, y, z):
		return np.full(x.shape, CY0)

	################ Y1 ################
	def Y1_1(x, y, z):
		return CY0 * y / r	

	def Y1(x, y, z):
		return CY0 * z / r	

	def Y11(x, y, z):
		return CY0 * x / r	

	################ Y2 ################
	def Y2_2(x, y, z):
		return CY2_2 * (x*y) / C_r_squared

	def Y2_1(x, y, z):
		return CY2_1 * (y*z) / C_r_squared	

	def Y2(x, y, z):
		return CY2 * (-x*x - y*y + 2*z*z) / C_r_squared	

	def Y21(x, y, z):
		return CY21 * (z * x) / C_r_squared	

	def Y22(x, y, z):
		return CY22 * (x*x - y*y) / C_r_squared	
		
		
	################ Y3 ################
	def Y3_3(x, y, z):
		return CY3_3 * (3*x*x - y*y) * y / C_r_cubed	

	def Y3_2(x, y, z):
		return CY3_2 * (x*y*z) / C_r_cubed	

	def Y3_1(x, y, z):
		return CY3_1 * (4*z*z - x*x - y*y) * y / C_r_cubed	

	def Y3(x, y, z):
		return CY3 * (2*z*z - 3*x*x - 3*y*y) * z / C_r_cubed	

	def Y31(x, y, z):
		return CY31 * (4*z*z - x*x - y*y) * x / C_r_cubed	

	def Y32(x, y, z):
		return CY32 * (x*x - y*y) * z / C_r_cubed	

	def Y33(x, y, z):
		return CY33 * (x*x - 3*y*y) * x / C_r_cubed	
		
	def computeCoefficentsY3(x, y, z):
		return np.array([Y3_3(x, y, z), Y3_2(x, y, z), Y3_1(x, y, z), Y3(x, y, z), Y31(x, y, z), Y32(x, y, z), Y33(x, y, z)])
		
	def computeY3(CEs, x, y, z):
		return Y3_3(x, y, z)*CEs[0] +	Y3_2(x, y, z)*CEs[1] + 	Y3_1(x, y, z)*CEs[2] + 	Y3(x, y, z)*CEs[3] +Y31(x, y, z)*CEs[4] +	Y32(x, y, z)*CEs[5] + Y33(x, y, z)*CEs[6]
		
	def computeCoefficentsY2(x, y, z):
		return np.array([Y2_2(x, y, z), Y2_1(x, y, z), Y2(x, y, z), Y21(x, y, z), Y22(x, y, z)])
		
	def computeY2(CEs, x, y, z):
		return Y2_2(x, y, z)*CEs[0] + 	Y2_1(x, y, z)*CEs[1] + 	Y2(x, y, z)*CEs[2] +Y21(x, y, z)*CEs[3] +	Y22(x, y, z)*CEs[4]
		
	def computeCoefficentsY1(x, y, z):
		return np.array([Y1_1(x, y, z), Y1(x, y, z), Y11(x, y, z)])
		
	def computeY1(CEs, x, y, z):
		return Y1_1(x, y, z)*CEs[0] + 	Y1(x, y, z)*CEs[1] +Y11(x, y, z)*CEs[2]
		
	def computeCoefficentsY0(x, y, z):
		return np.array([Y0(x, y, z)])
		
	def computeY0(CEs, x, y, z):
		return Y0(x, y, z)*CEs[0]
		
	Y0Table = [Y0]
	Y1Table = [Y1_1, Y1, Y11]
	Y2Table = [Y2_2, Y2_1, Y2, Y21, Y22]
	Y3Table = [Y3_3, Y3_2, Y3_1, Y3, Y31, Y32, Y33]
	YTable = [Y0Table, Y1Table, Y2Table, Y3Table]
		
	def getY(l, m):
		return YTable[l][m+l]
		

			
def vsh(l = 3, m = -3):				#Visualizing the spherical harmonics		(real)
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

	sh = getY(l, m)(x, y, z)
	#sh = Y0(x, y, z)
	#sh = Y1(x, y, z)

	sh = np.sqrt(2) * (-1)**m * sh.real
	
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
	
	
def project():
	
	
## https://www.chebfun.org/examples/sphere/SphericalHarmonics.html#:~:text=A%20spherical%20harmonic%20projection%20gives,periodic%20function%20in%20one%20dimension.
def cshc_band3():				#compute spherical harmonics coefficients
	global x, y, CEsn
	
	r = np.pi*1
	
	x = np.arange(-r, r, 0.1)
	fy = np.sin(x)
	#fy = x*x
	x = x / r		#normalize x
	y = 1. - x*x
	
	plt.plot(x, fy)
	#return
	
	i = 0
	CEs3 = np.zeros(7)
	for v in x:
		CEs3 = CEs3 + fy[i] * computeCoefficentsY3(v, y[i], 0)
		i = i+1
		
	CEs3 = CEs3 / len(x)
	
	
	i = 0
	CEs2 = np.zeros(5)
	for v in x:
		CEs2 = CEs2 + fy[i] * computeCoefficentsY2(v, y[i], 0)
		i = i+1
		
	CEs2 = CEs2 / len(x)	
	
	i = 0
	CEs1 = np.zeros(3)
	for v in x:
		CEs1 = CEs1 + fy[i] * computeCoefficentsY1(v, y[i], 0)
		i = i+1
		
	CEs1 = CEs1 / len(x)	
	
	i = 0
	CEs0 = np.zeros(1)
	for v in x:
		CEs0 = CEs0 + fy[i] * computeCoefficentsY0(v, y[i], 0)
		i = i+1
		
	CEs0 = CEs0 / len(x)	
	
	#x = x / r
	
	i = 0
	for v in x:
		y[i] = computeY3(CEs3, v, y[i], 0)
		i = i + 1
		
	i = 0
	for v in x:
		y[i] = y[i] + computeY2(CEs2, v, y[i], 0)
		i = i + 1

	i = 0
	for v in x:
		y[i] = y[i] + computeY1(CEs1, v, y[i], 0)
		i = i + 1

	i = 0
	for v in x:
		y[i] = y[i] + computeY0(CEs0, v, y[i], 0)
		i = i + 1

	y = y * 10
		
	plt.plot(x, y)
	plt.show()
	
	
