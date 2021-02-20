def u(x):
	if x<0 :
		return 0
	return 1

def pulse(x):
	cnt = len(t)
	y = np.zeros(cnt)
	for i,v in np.ndenumerate(t):
		if v < 0:
			y[i] = 0
		elif v > 2:
			y[i] = 0
		else:
			y[i] = 2
	return y


def negx(t):
	cnt = len(t)
	y = np.zeros(cnt)
	for i,v in np.ndenumerate(t):
		if v < 0:
			y[i] = 0
		elif v > 2:
			y[i] = 0
		else:
			y[i] = 2-v
	return y

def conv_negx_const(t):
	cnt = len(t)
	y = np.zeros(cnt)
	for i,v in np.ndenumerate(t):
		if v < 0:
			y[i] = 0
		elif v <= 2:
			y[i] = 4*v - v*v
		elif v <= 4:
			y[i] = 16 - 8*v + v*v
		else:
			y[i] = 0
	return y	
	

def pf(df, xstart, xstop, xnum = 50, color = 'green', show = True, linewidth=1.0):		
	global plt, t, s, xstep
	
	xstep = (xstop-xstart)/(xnum)
	
	t = np.arange(xstart, xstop+xstep, xstep)			#use xstop+xstep to include the endpoint.
	s = df(t)
	line, = plt.plot(t, s, color = color, linewidth = linewidth)		
	
	if show:
		plt.show()
		
	
def test_negx_pulse():
	pf(negx, -1, 5, 50, 'green', False)
	pf(pulse, -1, 5, 50, 'blue', False)
	pf(conv_negx_const, -1, 5, 250, 'red', True)	
	
	
