import json
import numpy

x = numpy.random.random((2000,1000))*2 - 1
w_opt = numpy.arange(1000,dtype=float)/10 - 50
b_opt = 1.0
y = numpy.random.random()*2 -1 + numpy.dot(x,w_opt) + b_opt
w = numpy.random.random(1000) * 100 - 50
b = numpy.random.random()
dicti = {'x':x.tolist(),'y':y.tolist(),'w':w.tolist(),'b':b}
dicti = json.dumps(dicti,indent=4)
with open("lasso.data","w") as fw:
	fw.write(dicti)

"""
optimum:
w = (-50.0, ... ,49.9)
b = 1.0

given:
x ~ [-1, 1)
"""