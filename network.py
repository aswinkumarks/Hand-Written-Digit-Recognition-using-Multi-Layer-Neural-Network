import numpy as np


def sigmoid(x):
	return np.exp(x)/(1+np.exp(x))

def delta_sigmoid(x):
	return sigmoid(x)*(1-sigmoid(x)) 
	# return x*(1-x)

def tanH(x):
	return np.tanh(x)

def delta_tanH(x):
	return 1.0 - x**2

activation_functions = {'sigmoid':sigmoid,'tanh':tanH}
delta_activation_functions = {'sigmoid':delta_sigmoid,'tanh':delta_tanH}

class MultilayerNeuralNetwork:

	def __init__(self,layers_dims,layer_activation_func):
		assert(len(layers_dims) >= 2)

		print('Using NOVA Network Backend\n Developed by Aswin Kumar\n')
		
		# list of activation functions used in each layer
		# eg : ['tanh','sigmoid']
		self.layer_activation_func = layer_activation_func

		# a list representing no of units in each layer
		# The length of this list is the no of layers in the network 
		# eg: No input neurons = 2
		#	  No of hidden layer units = 4
		#     No ouput neurons = 2
		#	  [2,4,2]
		self.layers_dims = layers_dims

		self.weights = []
		self.biases = []

		for i in range(1,len(layers_dims)):
			weight = np.random.randn(layers_dims[i], layers_dims[i-1])*0.01
			biase = np.zeros((layers_dims[i], 1))
			
			self.weights.append(weight)
			self.biases.append(biase)

	def forward(self,inputs):
		acti = np.copy(inputs)
		activations = [acti]
		Z_S = []

		for i in range(len(self.weights)):
			z = self.weights[i].dot(acti) + self.biases[i]
			Z_S.append(z)

			acti = activation_functions[self.layer_activation_func[i]](z)
			activations.append(acti)

		return Z_S,activations


	def predict(self,inputs):
		acti = inputs
		for i in range(len(self.weights)):
			z = self.weights[i].dot(acti) + self.biases[i]

			acti = activation_functions[self.layer_activation_func[i]](z)

		return acti

	def compute_cost(self,Y,A):
		'''
			Y : actual target values
			A : predicted values
		'''

		m = Y.shape[1]
		# Cross entopy loss
		cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
		cost = np.squeeze(cost)

		return cost

	def backpropagation(self,Y,Z_S,activations):
		'''
			Y : actual target values
			Z_S : values before activation is applied
			activations : values after activation is applied
		'''
		deltas = [None] * len(self.weights)
		# Output layer delta value
		dAL = - (np.divide(Y, activations[-1]) - np.divide(1 - Y, 1 - activations[-1]))  
		deltas[-1] = dAL * delta_activation_functions[self.layer_activation_func[-1]](Z_S[-1])
		# Calculation of hidden layer delta values
		for i in reversed(range(len(deltas)-1)):
			deltas[i] = self.weights[i+1].T.dot(deltas[i+1]) \
						* delta_activation_functions[self.layer_activation_func[-1]](Z_S[i])        

		batch_size = Y.shape[1]
		# 
		db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
		dw = [d.dot(activations[i].T)/float(batch_size) for i,d in enumerate(deltas)]
		# print(db,dw)
		# exit(0)
		return dw, db

	def train(self, x, y, batch_size=1000, epochs=100, lr = 0.001,model_name='model_weights.npz'):
		for e in range(1,epochs): 
			i=0
			print("\nEpoch %d"%(e))
			while(i<y.shape[1]):
				x_batch = x[:,i: i + batch_size]
				y_batch = y[:,i: i + batch_size]
				i = i + batch_size
				z_s, a_s = self.forward(x_batch)
				dw, db = self.backpropagation(y_batch, z_s, a_s)
				self.weights = [ w - lr * dweight for w,dweight in  zip(self.weights, dw)]
				self.biases = [ w - lr * dbias for w,dbias in  zip(self.biases, db)]
				print("loss = {}".format(self.compute_cost(y_batch,a_s[-1])),end='\r')
				
			self.save_weights(model_name)

		print('\n')

	def save_weights(self,filename='model_weights.npz'):
		np.savez(filename, layers_dims = self.layers_dims,
			     weights = self.weights, biases=self.biases,
			     layer_activation_func = self.layer_activation_func)

	def load_weights(self,filename='model_weights.npz'):
		try:
			npzfile = np.load(filename,allow_pickle=True)
			self.layers_dims = npzfile['layers_dims']
			self.weights = npzfile['weights']
			self.biases = npzfile['biases']
			self.layer_activation_func = npzfile['layer_activation_func']

		except:
			print('Error loading weights')
			exit(0)		
