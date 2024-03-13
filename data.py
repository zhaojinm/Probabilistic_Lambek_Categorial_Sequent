from bs4 import BeautifulSoup
# import seaborn as sns
# import matplotlib.pyplot as plt
from node import *
import math

POS="+"
NEG='-'
cur_label=0

class lctreenode:
	def __init__(self, con, prim):
		if "_" in prim:
			self.prim = prim[:prim.find('_')]
		else:
			self.prim = prim
		self.con = con
		self.child = []
		self.par = None
		self.name = con+self.prim
	def count_prim(self):
		count = 0
		if self.prim!="L":
			count = 1
		for c in self.child:
			count += c.count_prim()
		# print("prim", count)
		return count
	def count_lambda(self):
		count = 0
		if self.prim=="L":
			count = 1
		for c in self.child:
			count += c.count_lambda()
		# print("lambda:",count)
		return count
	def count_cat(self):
		return self.count_prim() - self.count_lambda()
	def count_lambda_subtree_cat_equal_one(self):
		cur = 0
		if self.prim=="L":
			# print("L")
			cur = int(self.count_cat()==1)
		for c in self.child:
			cur += c.count_lambda_subtree_cat_equal_one()
		return cur
	def __str__(self,level=0,cidx=0):
		res = ""
		if cidx==0:
			if self.child!=[]:
				res = res + "("+ self.name + "\t"
			else:
				res = res +  self.name
		else:
			if self.child!=[]:
				res = res+"\n" +"\t"*level + "("+self.name+"\t"
			else:
				res = res+"\n" +"\t"*level + self.name+"\t"
		for (i,c) in enumerate(self.child):
			res+=c.__str__(level+1,i)
			if c.child!=[]:
				res+=")"
		if level==0:
			res+=")"
		return res
	
def decrement_label():
	global cur_label
	cur_label-=1

def get_cur_lable():
	global cur_label
	# ch = chr(ord('a')+cur_label)
	ch = str(cur_label)
	cur_label+=1
	return ch

def reset_cur_label():
	global cur_label
	cur_label = 0

def generate_node(cat,sign,label, lambda_pairs, label_to_tree_prim, con=" "):
	"str->Node"
	if not ("\\" in cat or "/" in cat):
		if sign == POS:
			label_to_tree_prim[label] = [con, cat]
		return Node(cat,sign,label)
	else:
		left_p=0
		for i,c in enumerate(cat):
			if c=="(":
				left_p+=1
			elif c==")":
				left_p-=1
			elif (c=="/" or c=="\\") and left_p==0:
				left = cat[:i]
				if "/" in left or "\\" in left:
					left = left[1:-1]
				right = cat[i+1:]
				if "/" in right or "\\" in right:
					right = right[1:-1]
				break
		if c=="/" and sign==POS:
			left_node=generate_node(right,NEG,get_cur_lable(), lambda_pairs, label_to_tree_prim)
			right_node=generate_node(left,POS,get_cur_lable(), lambda_pairs,label_to_tree_prim,'/')
			lambda_pairs[label] = left_node.label+" "+right_node.label
			label_to_tree_prim[label] = [con, 'L']
		elif c=="/" and sign==NEG:
			right_node = generate_node(right,POS,get_cur_lable(), lambda_pairs, label_to_tree_prim,"/")
			left_node=generate_node(left,NEG,label+" "+right_node.label, lambda_pairs,label_to_tree_prim)
		elif c=="\\" and sign==POS:
			left_node=generate_node(left,POS,get_cur_lable(), lambda_pairs, label_to_tree_prim, '\\')
			right_node=generate_node(right,NEG,get_cur_lable(), lambda_pairs,label_to_tree_prim)
			lambda_pairs[label] = left_node.label+" "+right_node.label
			label_to_tree_prim[label] = [con, 'L']
		elif c=="\\" and sign==NEG:
			left_node=generate_node(right,POS, get_cur_lable(), lambda_pairs, label_to_tree_prim,"\\")
			right_node=generate_node(left,NEG, label+" "+left_node.label, lambda_pairs, label_to_tree_prim)
		return Node(pol=sign,label=label,left=left_node,right=right_node,op=c)

def generate_proofnet(cat, match):
	folding = []
	unfolding = []
	lambda_pairs = {}
	label_to_tree_prim = {}
	for c in cat[:-1]:
		cur_cat_node = generate_node(c,NEG,get_cur_lable(), lambda_pairs, label_to_tree_prim)
		folding.append(cur_cat_node)
		unfolding+=cur_cat_node.leaf()
	cur_cat_node=generate_node(cat[-1],POS,get_cur_lable(), lambda_pairs, label_to_tree_prim)
	folding.append(cur_cat_node)
	unfolding+=cur_cat_node.leaf()
	for m in match:
		unfolding[m[0]-1].add_link(unfolding[m[1]-1])
	return (folding, unfolding, lambda_pairs, label_to_tree_prim)

def build_tree_from_proofnet(label_to_tree_prim,pairs,cur):
	cur_node = lctreenode(label_to_tree_prim[cur][0],label_to_tree_prim[cur][1])
	if len(pairs[cur].split())==1:
		return cur_node
	for c in pairs[cur].split():
		if c in label_to_tree_prim:
			c_node = build_tree_from_proofnet(label_to_tree_prim, pairs, c)
			cur_node.child.append(c_node)
			c_node.par = cur_node
	return cur_node
	
def proofnet_2_tree(cats, matches):
	'''
	'''
	# print(cats)
	# print(matches)
	# exit()
	global cur_label
	cur_label = 0
	folding, unfolding, pairs, label_to_tree_prim = generate_proofnet(cats, matches)
	# print(folding, unfolding, pairs, label_to_tree_prim)
	root = folding[-1].label
	# print(root)
	for u in unfolding:
		if u.pol==POS:
			pairs[u.label] = u.link.label
	# print(pairs)
	root_node = build_tree_from_proofnet(label_to_tree_prim, pairs,root)
	# print(root_node)
	return root_node
def print_cat(cur_cat):
	for c in cur_cat[:-1]:
		cc = ''.join([i for i in c if not (i.isdigit() or i=="_")])
		print(cc,end = " ")
	for c in cur_cat[-1:]:
		cc = ''.join([i for i in c if not (i.isdigit() or i=="_")])
		print(cc)
def print_word(cur_word):
	for c in cur_word[:-1]:
		print(c,end = " ")
	for c in cur_word[-1:]:
		print(c)
def load_trees(folder='tst',primitives = ["S","NP","N","PP","conj"]):
	word_to_cat = {}
	cat_to_word = {}
	all_trees = []
	all_words = []
	all_matches = []

	with open('./data/LCGbank.'+folder+'.xml') as f:
		data = f.read()
	_data = BeautifulSoup(data,"xml")
	right_ = _data.find_all("sentential")
	words = _data.find_all("words")
	matching = _data.find_all("matching")

	for ws,mt,r in zip(words[:],matching[:],right_[:]):
		cur_cat = []
		cur_match = []
		cur_word = []
		for w in ws.find_all("word"):
			try:
				c = w["cat"]
				text = w["text"].lower()
				if text in word_to_cat:
					word_to_cat[text].append(''.join([i for i in c if (not i.isdigit()) and i!="_"]))
				else:
					word_to_cat[text] = [''.join([i for i in c if (not i.isdigit()) and i!="_"])]
				if ''.join([i for i in c if (not i.isdigit()) and i!="_"]) in cat_to_word:
					cat_to_word[''.join([i for i in c if (not i.isdigit()) and i!="_"])].append(text)
				else:
					cat_to_word[''.join([i for i in c if (not i.isdigit()) and i!="_"])] = [text]
				cur_cat.append(c)
				cur_word.append(w["text"].lower())
			except:
				pass
		for m in mt.find_all("match"):
			cur_match.append((int(m["negative"]),int(m["positive"])))
		assert len(cur_cat)==len(cur_word)
		# print_cat(cur_cat)
		# print_word(cur_word)
		cur_cat.append(r["cat"])
		tree = proofnet_2_tree(cur_cat,cur_match)
		#print(tree)
		#print(tree.count_prim(), tree.count_lambda(), tree.count_cat())
		all_trees.append(tree)
		all_words.append(cur_word)
		all_matches.append(cur_match)
	print("---load {} set is done with {} sequents----".format(folder,len(all_trees)))
	return all_trees,all_words,all_matches

def load_standard_word(folder='trn'):
	with open('./data/LCGbank.'+folder+'.xml') as f:
		data = f.read()
	_data = BeautifulSoup(data,"xml")
	right_ = _data.find_all("sentential")
	words = _data.find_all("words")
	matching = _data.find_all("matching")
	allwords = {}
	for ws,mt,r in zip(words[:],matching[:],right_[:]):
		for w in ws.find_all("word"):
			try:
				c = w["cat"]
				text = w["text"].lower()
				if text.replace('.','').replace('-',"").replace("/","").isnumeric() or text=='%':
					text = 'N'
				if text in allwords:
					allwords[text]+=1
				else:
					allwords[text]=1
			except:
				pass
	standard_words = [w for w in allwords if allwords[w]>=6]
	print(len(standard_words))
	# print(standard_words)
	return standard_words

def get_cat_word_mapping(standard_voc, folder = 'trn'):
	cat_to_word = {}
	word_to_cat = {}
	with open('./data/LCGbank.'+folder+'.xml') as f:
		data = f.read()
	_data = BeautifulSoup(data,"xml")
	right_ = _data.find_all("sentential")
	words = _data.find_all("words")
	matching = _data.find_all("matching")

	for ws,mt,r in zip(words[:],matching[:],right_[:]):
		for w in ws.find_all("word"):
			try:
				c = w["cat"]
				c = ''.join([i for i in c if (not i.isdigit()) and i!="_"])
				text = w["text"].lower()
				if text.replace('.','').replace('-',"").replace("/","").isnumeric() or text=='%':
					text = 'N'
				if text not in standard_voc:
					text = '<unk>'
				if text in word_to_cat:
					word_to_cat[text].append(c)
				else:
					word_to_cat[text] = [c]
				if c in cat_to_word:
					cat_to_word[c].append(text)
				else:
					cat_to_word[c] = [text]
			except:
				pass
	return cat_to_word, word_to_cat

def count_words(standard_voc, cat_to_word, folder='tst'):
	total_word = 0
	total_p = 0
	with open('./data/LCGbank.'+folder+'.xml') as f:
		data = f.read()
	_data = BeautifulSoup(data,"xml")
	right_ = _data.find_all("sentential")
	words = _data.find_all("words")
	matching = _data.find_all("matching")
	allwords = {'<unk>':0}
	for ws,mt,r in zip(words[:],matching[:],right_[:]):
		for w in ws.find_all("word"):
			try:
				c = w["cat"]
				text = w["text"].lower()
				c=''.join([i for i in c if (not i.isdigit()) and i!="_"])
				if text.replace('.','').replace('-',"").replace("/","").isnumeric() or text=='%':
					text = 'N'
				if text in standard_voc:
					if text in allwords:
						allwords[text]+=1
					else:
						allwords[text] = 1
				else:
					text = '<unk>'
					allwords[text]=0
				print(text)
				total_p+=math.log(float(cat_to_word[c].count(text))/float(len(cat_to_word[c])))
				total_word+=1
			except:
				pass
	print(total_p, total_word)
	print(len(allwords.keys()))

if __name__=='__main__':
	trees = load_trees("dev")
	# print(len(trees))
	# print(trees[-1])
	# standard_voc = load_standard_word('trn')
	# cat_to_word, word_to_cat = get_cat_word_mapping(standard_voc,'trn')
	# print("total words in trn: ", sum([len(cat_to_word[c]) for c in cat_to_word]))
	# print(len(cat_to_word['NP/NP']),len(word_to_cat))
	# count_words(standard_voc,cat_to_word,'tst')