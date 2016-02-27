#coding:utf-8
import numpy
import json

class Lasso:
	def __init__(self,lam=10**-3,stepsize=10**-3,sample_size=10,\
			dim_size=10):
		self.x = numpy.array([[1,1],[1,1],[1,1],[2,2]],dtype=float)
		self.y = numpy.array([1,1,1,3],dtype=float)
		self.w = numpy.array([1,1],dtype=float)
		self.b = 1
		self.l = 0

		self.lam = lam
		self.stepsize = stepsize
		self.sample_size = sample_size
		self.dim_size = dim_size

	def ff(self):
		f_theta = 0.5 * (numpy.dot(self.x,self.w) + self.b)**2
		self.l = numpy.mean(f_theta) + self.lam * numpy.sum(numpy.abs(self.w))
		return self.l

	def bp_and_update(self,step=1):
		decayedstep = 1000.0/(1000.0+step) * self.stepsize

		n = self.x.shape[0]
		d = self.x.shape[1]
		i = numpy.random.randint(n)
		j = numpy.random.randint(d)

		for jj in range(self.dim_size):
			l_w = 0
			for ii in range(self.sample_size):
				l_w += (numpy.dot(self.w,self.x[(i+ii)%n]) + self.b - self.y[(i+ii)%n])\
					 * self.x[(i+ii)%n,(j+jj)%d]
			l_w = float(l_w)/self.sample_size
			if self.w[(j+jj)%d] > 0:
				l_w += self.lam
			elif self.w[(j+jj)%d] < 0:
				l_w -= self.lam
			else:
				pass
			self.w[(j+jj)%d] -= decayedstep * l_w

		l_b = 0
		for ii in range(self.sample_size):
			l_b += numpy.dot(self.w,self.x[(i+ii)%n]) + self.b - self.y[(i+ii)%n]
		l_b = float(l_b)/self.sample_size
		self.b -= decayedstep * l_b

if __name__ == "__main__":
	model = Lasso()
	with open("lasso.data","r") as fr:
		d = json.load(fr)
		model.b = d['b']
		model.x = numpy.array(d['x'])
		model.y = numpy.array(d['y'])
		model.w = numpy.array(d['w'])
	print model.ff()
	for i in range(100):
		model.bp_and_update()
		if i % 10 == 0 :
			print model.ff()


