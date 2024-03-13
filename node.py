class Node():
	def __init__(self,prime=None,pol=None,label=None,op=None,left=None,right=None):
		self.cat = prime
		self.pol = pol
		self.label = label
		self.left = left
		self.right = right
		if left!=None and right!=None:
			self.child = [left,right]
			left.parent=self
			right.parent=self
		else:
			self.child = []
		self.parent = None
		self.op = op
		self.link=None
		self.lcnode=None
	def count(self):
		return len(self.leaf())
	def leaf(self):
		if self.child ==[]:
			return [self]
		else:
			return self.left.leaf()+self.right.leaf()
	def find_leaf_from_positive_path(self):
		if self.left==None and self.right==None and self.pol=="+":
			return self.cat
		if self.left.pol=="+":
			return self.left.find_leaf_from_positive_path()
		else:
			return self.right.find_leaf_from_positive_path()
	def add_link(self,c):
		self.link=c
		c.link=self
	def updatechild_label(self,ori,des):
		# print(ori,des)
		if self.left!=None:
			if ori in self.left.label:
				self.left.label=self.left.label.replace(ori,des)
				self.left.updatechild_label(ori,des)
			if ori in self.right.label:
				self.right.label=self.right.label.replace(ori,des)
				self.right.updatechild_label(ori,des)
	def __str__(self):
		if self.op==None:
			return self.cat+":"+self.label
		else:
			left_s = self.left.__str__()
			right_s = self.right.__str__()
			if self.left.op!=None:
				left_s='('+left_s+')'
			if self.right.op!=None:
				right_s='('+right_s+')'
			if self.pol=="+":
				return right_s+self.op+left_s+":"+self.label
			return left_s+self.op+right_s+":"+self.label
	def __str__1(self):
		if self.op==None:
			return self.cat+self.pol+self.label
		else:
			left_s = self.left.__str__()
			right_s = self.right.__str__()
			if self.left.op!=None:
				left_s='('+left_s+')'
			if self.right.op!=None:
				right_s='('+right_s+')'
			return left_s+self.op+right_s+" "+self.label
	def __repr__(self):
		return self.__str__()
class LcNode():
	def __init__(self,node=None,islambda=False,islambdaplus=False):
		assert len(node.label)==1
		self.label=node.label
		self.node=node
		self.child = []
		self.parent = [] 
		self.islambda=islambda
		self.islambdaplus=islambdaplus
		node.lcnode=self
	def add_child(self,c):
		self.child.append(c)
		c.parent.append(self)
	def del_child(self,c):
		self.child.remove(c)
		c.parent.remove(self)
	def __str__(self):
		result=""
		if self.child!=[]:
			result = self.label + "=" + "".join([c.label for c in self.child])+" "
			for c in self.child:
				result = result + c.__str__()+ " "
		return result
	def __repr__(self):
		return self.__str__()
	def get_leaf(self):
		result=set()
		if self.child==[]:
			result.add(self)
		for c in self.child:
			result=result|c.get_leaf()
		return result

	def get_possible_pairs(self):
		result = []
		if self.child!=[]:
			plus=self
			leaf = self.get_leaf()
			# if sum([int(len(l.parent)<=1) for l in leaf])<=1:
			# 	return result
			for l in leaf:
				if len(l.parent)==1:
					for ll in leaf:
						parents = ll.parent
						if sum([int(p.islambdaplus or p==self) for p in parents])==0:
							result.append((self,l))
							break
			if not self.islambda:
				for c in self.child:
						result+=c.get_possible_pairs()
		return result