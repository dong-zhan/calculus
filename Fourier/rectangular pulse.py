def unit_step_fn(x):
	global T
	cnt = len(x)
	y = np.zeros(cnt)
	for i,v in np.ndenumerate(x):
		if v>= -1 and v<=1:
			y[i] = 1
		else:
			y[i] = 0
	return y

def rectangular_pulse(inT, interval = np.pi/30):
	global x, T, y
	T = inT
	ext = T*0.2
	x = np.arange(-T-ext, T+ext+ext*0.1, interval)
	y = unit_step_fn(x)
	plt.plot(x, y)
	
	plt.show()
	
def test_fft(ndArr, spacing):
	global X, Y
	samples = int(len(ndArr))
	Y = scipy.fftpack.fft(ndArr)
	X = np.linspace(0.0, 1.0/(2.0*spacing), int(samples/2))

	plt.plot(X, 2.0/samples * np.abs(Y[:int(samples//2)]))
	plt.show()
	
def fourier_unit_step_fn(x):
	global T
	return 2*np.sin(x*T)/x
	
def fourier_rectangular_pulse(inT, interval = np.pi/30):
	global x, T, y
	T = inT
	ext = T*0.2
	x = np.arange(-T-ext, T+ext+ext*0.1, interval)
	y = fourier_unit_step_fn(x)
	plt.plot(x, y)
	
	plt.show()	
	
rectangular_pulse(10)
test_fft(y, 0.1)
fourier_rectangular_pulse(10)
