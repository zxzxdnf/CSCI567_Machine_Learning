import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		h = np.zeros(len(features))
		for clf, beta in zip(self.clfs_picked, self.betas):	
			h = h + beta * np.array(clf.predict(features))

		h = [-1 if hn <= 0 else 1 for hn in h]
		return h
 		
		########################################################
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################

		N = len(labels)
		D = np.full(N, 1 / N)
	
		for t in range(self.T):

			ep = float("inf")
			for c in self.clfs:
				h = c.predict(features)
				er = np.sum(D * (np.array(labels) != np.array(h)))
				if er < ep:
					ht = c
					ep = er
					htx = h
			self.clfs_picked.append(ht)
			

			beta = 1 / 2 * np.log((1 - ep) / ep)
			self.betas.append(beta)

			for n in range(N):
				if labels[n] == htx[n]:
					D[n] *= np.exp(-beta)
				else:
					D[n] *= np.exp(beta)


			D /= np.sum(D)
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	