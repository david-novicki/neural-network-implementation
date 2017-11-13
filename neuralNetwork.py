import numpy as np
import scipy.special as sc
class neuralNetwork:
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learningrate
		self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.inodes))
		self.activation_function = lambda x: sc.expit(x)
		pass

	def train(self, inputs_list, targets_list):
		
		pass
	
	def query(self, inputs_list):
		#convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T
		#calculate the signals into the hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		#calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)
		#calculate signals into the final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		#calculate the signals emerging from the final output
		final_outputs = self.activation_function(final_inputs)
		return final_outputs

pass

	