import numpy
import json

class LassoIII:
	def __init__(self,lam=10**-3,stepsize=10**-3,sample_size=10,\
			dim_size=10,frequency=10,lowerbound=10**-3):
		self.x = numpy.array([[1,1],[1,1],[1,1],[2,2]],dtype=float)
		self.y = numpy.array([1,1,1,3],dtype=float)
		self.w = numpy.array([1,1],dtype=float)
		self.b = 1.0
		self.l = 0

		self.lam = lam
		self.stepsize = stepsize
		self.sample_size = sample_size
		self.dim_size = dim_size
		self.frequency = frequency
		self.lowerbound = lowerbound

	def ff(self):
		f_theta = 0.5 * (numpy.dot(self.x,self.w) + self.b)**2
		self.l = numpy.mean(f_theta) + self.lam * numpy.sum(numpy.abs(self.w))
		return self.l

	def bp_and_update(self):
		n = self.x.shape[0]
		d = self.x.shape[1]

		mu_w = numpy.zeros(self.w.shape,dtype=float)
		mu_b = 0.0
		for ii in range(n):
			mu_w += (numpy.dot(self.w,self.x[ii]) + self.b - self.y[ii])\
				* self.x[ii]
			mu_b += numpy.dot(self.w,self.x[ii]) + self.b - self.y[ii]
		mu_w = mu_w/n
		mu_b = float(mu_b)/n

		activeset = set([])
		for k in range(d):
			if mu_w[k] >= self.lowerbound:
				activeset.add(k)

		w2 = numpy.array(self.w)
		b2 = self.b
		for t in range(self.frequency):
			i = numpy.random.randint(n)
			j = numpy.random.randint(d)

			for jj in range(self.dim_size):
				if activeset.__contains__(jj) == False:
					continue
				l_w = 0
				l_w2 = 0
				for ii in range(self.sample_size):
					l_w += (numpy.dot(self.w,self.x[(i+ii)%n]) + self.b - self.y[(i+ii)%n])\
						 * self.x[(i+ii)%n,(j+jj)%d]
					l_w2 += (numpy.dot(w2,self.x[(i+ii)%n]) + b2 - self.y[(i+ii)%n])\
						 * self.x[(i+ii)%n,(j+jj)%d]
				l_w = float(l_w)/self.sample_size
				l_w2 = float(l_w2)/self.sample_size
				l_sumw = l_w2 - l_w + mu_w[(j+jj)%d]
				if self.w[(j+jj)%d] > 0:
					l_sumw += self.lam
				elif self.w[(j+jj)%d] < 0:
					l_sumw -= self.lam
				else:
					pass
				w2[(j+jj)%d] -= self.stepsize * l_sumw

			l_b = 0
			l_b2 = 0
			for ii in range(self.sample_size):
				l_b += numpy.dot(self.w,self.x[(i+ii)%n]) + self.b - self.y[(i+ii)%n]
				l_b2 += numpy.dot(w2,self.x[(i+ii)%n]) + b2 - self.y[(i+ii)%n]
			l_b = float(l_b)/self.sample_size
			l_b2 = float(l_b2)/self.sample_size
			l_sumb = l_b2 - l_b + mu_b
			b2 -= self.stepsize * l_sumb
		self.w = w2
		self.b = b2

if __name__ == "__main__":
	model = LassoIII()
	with open("lasso.data","r") as fr:
		d = json.load(fr)
		model.b = d['b']
		model.x = numpy.array(d['x'])
		model.y = numpy.array(d['y'])
		model.w = numpy.array(d['w'])
	print "model.ff() = ",model.ff()
	for i in range(100):
		model.bp_and_update()
		if i % 10 == 0 :
			print "model.ff() = ",model.ff()