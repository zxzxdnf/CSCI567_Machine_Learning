import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			BR = np.array(branches)
			TOT = np.sum(BR, axis=0)
			FR = TOT / np.sum(TOT)
			EN = BR / TOT
			EN = np.array([[-j * np.log2(j)  if j > 0 else 0 for j in i] for i in EN])
			EN = np.sum(EN, axis=0)
			EN = np.sum(EN * FR)
			return EN			
		
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			if not 'ME' in locals():
				ME = float('inf')
			WW = np.array(self.features)[:, idx_dim]
			if None in WW:
				continue
			BRV = np.unique(WW)
			BR = np.zeros((self.num_cls, len(BRV)))
			for i, val in enumerate(BRV):
				y = np.array(self.labels)[np.where(WW == val)]
				for yi in y:
					BR[yi, i] = BR[yi, i] + 1
			EN = conditional_entropy(BR)
			if EN < ME:
				ME = EN
				self.dim_split = idx_dim
				self.feature_uniq_split = BRV.tolist()



		############################################################
		# TODO: split the node, add child nodes
		############################################################
		AW = np.array(self.features)[:, self.dim_split]
		AQ = np.array(self.features, dtype = object)
		AQ[:, self.dim_split] = None

		for S in self.feature_uniq_split:
			
			IN = np.where(AW == S)
			
			YN = np.array(self.labels)[IN].tolist()

			XN = AQ[IN].tolist()

			child = TreeNode(XN, YN, self.num_cls)

			if np.array(XN).size == 0 or all(w is None for w in XN[0]):
				child.splittable = False

			self.children.append(child)



		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



