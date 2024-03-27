import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from bs4 import BeautifulSoup
# import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import sys
sys.path.append('../../')
from data import *
from parametrize_and_test import load_rule_from_tree
import argparse
import matplotlib.pyplot as plt
import numpy as np

'''
Main idea about bounded order:
The key idea is that for each entry fi,j that represents the matchings from Xi to Xj(inclusive),
we only care about it integral critira is satisfied for subgraph that contains nodes from root to
Xi and nodes from root to Xj only. Both braketing and adjoin is a function of m. Worst case is
adjoin that combined graph contains 4m nodes at most.

For T(CT), if A+ solid access to B- and A does not solid access to B' lambda parent then CT is satisfied
for A. So verify CT also can be done during braket and adjoin with complexity a function of m.

n^2 entry, each entry does 1 brackt and O(n) adjoin.  Overall O(n^3).
'''
def parse_arguments():
    parser = argparse.ArgumentParser(description="Probabilistic Lambek Categorial Sequent")

    parser.add_argument("--par_depth", type=int, default="2", help="depth of parant")
    parser.add_argument("--lambda_is_par", action='store_true', help="consider lambda node as parant node or not")
    
    args = parser.parse_args()
    
    return args

args = parse_arguments()
print(args)

POS = 1
NEG = 0
node_num = 0
train_trees,train_words,train_matches = load_trees('trn')
test_trees,test_words,test_matches = load_trees('tst')
print(len(train_trees))
count = 0
all_rule = {}
for t in train_trees[:]:#+dev_trees[:]:
    # print(t)
    rule = load_rule_from_tree(t,args)
    for r in rule:
        left,right = r
        if left in all_rule:
            if right in all_rule[left]:
                all_rule[left][right] += rule[r]
            else:
                all_rule[left][right] = rule[r]
        else:
            all_rule[left] = {right:rule[r]}
        count+=rule[r]

print()
# print(all_rule)
p_rule = {}
for left in all_rule:
    p = {}
    for right in all_rule[left]:
        p[right] = float(all_rule[left][right])/float(sum([all_rule[left][k] for k in all_rule[left]]))
    p_rule[left] = p

class UnfoldNode():
    def __init__(self, prim, num, sign) -> None:
        self.prim = prim
        self.num = num
        self.sign = sign
        self.right_child = []
        self.left_child = []
        self.par = None
        self.depth = 0

    def set_depth(self,depth):
        self.depth = depth
        for c in self.left_child+self.right_child:
            c.set_depth(depth+1)

    def __repr__(self) -> str:
        return self.prim + str(self.num) + " "+ str(self.sign)

    def __str__(self) -> str:
        return self.prim + str(self.num) + " "+ str(self.sign)
    def inorder(self):
        result = []
        for c in self.left_child:
            result.extend(c.inorder())
        result.append(self)
        for c in self.right_child:
            result.extend(c.inorder())
        return result

class Graph():
    def __init__(self,vertices,lambdaPairs,negativeNodes, postiveLambda):
        self.graph = defaultdict(set)
        self.V = vertices
        self.dashgraph = defaultdict(set)
        self.negativeNodes = negativeNodes
        self.lambdaPairs = lambdaPairs
        self.postiveLambda = postiveLambda
        self.negativePar = defaultdict(int)
        self.CT = set()
        self.matching = set()
        for p in lambdaPairs:
            self.negativePar[p[1]] = p[0]

    def addEdge(self,u,v):
        self.graph[u].add(v)

    def addDash(self,u,v):
        self.dashgraph[u].add(v)

    def isCyclicUtil(self, v, visited, recStack):
        visited[v] = True
        recStack[v] = True
        for neighbour in self.graph[v]:
            if visited[neighbour] == False:
                if self.isCyclicUtil(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                return True
        recStack[v] = False
        return False

    # Returns true if graph is cyclic else false
    def isCyclic(self):
        visited = dict()
        recStack = dict()
        for v in self.V:
            visited[v] = False
            recStack[v] = False
        for node in self.V:
            if visited[node] == False:
                if self.isCyclicUtil(node,visited,recStack) == True:
                    return True
        return False

    def isAccess(self, u, v):
        if u==v:
            return True
        visited = dict()
        for n in self.V:
            visited[n] = False
        queue = [u]
        while queue!=[]:
            cur = queue.pop(0)
            for neighbor in self.graph[cur]:
                if neighbor==v:
                    return True
                if visited[neighbor]!=visited:
                    queue.append(neighbor)
        return False

    def __str__(self) -> str:
        return str(self.graph)+str(self.dashgraph)+str(self.CT)+str(self.lambdaPairs)\
            +str(self.negativeNodes)+str(self.negativePar)+str(self.postiveLambda)+str(self.matching)+"\n"

    def __repr__(self) -> str:
        return str(self.graph)+str(self.dashgraph)+str(self.CT)+str(self.lambdaPairs)\
            +str(self.negativeNodes)+str(self.negativePar)+str(self.postiveLambda)+str(self.matching)+"\n"

    def isSame(self,other,empty_premises=True):
        if self.V!=other.V:
            return False
        for u in self.V:
            if self.graph[u]!=other.graph[u]:
                return False
            if self.dashgraph[u]!=other.dashgraph[u]:
                return False
        if not empty_premises:
            return self.CT==other.CT
        return True

def get_main_connective(cat):
    num_left_para=0
    for i,ch in enumerate(cat):
        if ch == "(":
            num_left_para+=1
        elif ch == ")":
            num_left_para-=1
        elif (ch=="/" or ch=="\\") and num_left_para==0:
            B = cat[i+1:]
            if B.count("/") + B.count("\\") >0:
                B = B[1:-1]
            A = cat[:i]
            if A.count("/") + A.count("\\") >0:
                A = A[1:-1]
            # print(ch,i,B,A)
            return ch, i, B, A
    return "", -1, "", ""

'''
A/B-: A- -->  B+
A\B-: B+  <-- A-
A/B+: B-  <--(dash) A+
A\B+: A+ -->(dash) B+
'''
def get_unfolding(c, sign):
    if c.count("/")+c.count("\\")==0:
        global node_num
        node_num += 1
        return UnfoldNode(c, node_num, sign)
    ch,i,B,A = get_main_connective(c)
    if ch=="/" and sign==NEG:
        A_node = get_unfolding(A,NEG)
        B_node = get_unfolding(B,POS)
        A_node.right_child.append(B_node)
        B_node.par = A_node
        return A_node
    elif ch=="\\" and sign ==NEG:
        B_node = get_unfolding(B,POS)
        A_node = get_unfolding(A,NEG)
        A_node.left_child.insert(0,B_node)
        B_node.par = A_node
        return A_node
    elif ch=="/" and sign==POS:
        B_node = get_unfolding(B,NEG)
        A_node = get_unfolding(A,POS)
        A_node.left_child.insert(0,B_node)
        B_node.par = A_node
        return A_node
    elif ch=="\\" and sign==POS:
        A_node = get_unfolding(A,POS)
        B_node = get_unfolding(B,NEG)
        A_node.right_child.append(B_node)
        B_node.par = A_node
        return A_node

def get_result_node(unfolding_formulae:list, s:int, e:int)->set:
    result = set()
    start_node = unfolding_formulae[s-1]
    end_node = unfolding_formulae[e-1]
    result_negativeNodes = set()
    result_lambda_pairs = set()
    start_path = []
    end_path = []
    result_lambda_nodes = set()

    if start_node.sign==POS and start_node.par!=None and \
        (start_node.left_child!=[] or start_node.right_child!=[]):
        result_lambda_nodes.add(start_node.num)
    if end_node.sign==POS and end_node.par!=None and \
        (end_node.left_child!=[] or end_node.right_child!=[]):
        result_lambda_nodes.add(end_node.num)

    while start_node:
        result.add(start_node.num)
        start_path.insert(0,start_node.num)
        start_node = start_node.par

    while end_node:
        result.add(end_node.num)
        end_path.insert(0,end_node.num)
        end_node = end_node.par

    for i,c in enumerate(start_path):
        if i%2==0 and i!=len(start_path)-1 and i!=0:
            result_lambda_nodes.add(c)
        if i%2==1:
            result_negativeNodes.add(c)
        if i%2==1 and i>=3:
            result_lambda_pairs.add((start_path[i-1], c))
    for i,c in enumerate(end_path):
        if i%2==0 and i!=len(end_path)-1 and i!=0:
            result_lambda_nodes.add(c)
        if i%2==1:
            result_negativeNodes.add(c)
        if i%2==1 and i>=3:
            result_lambda_pairs.add((end_path[i-1], c))

    return result, start_path, end_path, result_lambda_nodes, result_negativeNodes, result_lambda_pairs

def combine(graph1, graph2, result_nodes,result_negativeNodes, result_lambda_pairs,result_lambda_nodes,get_matching=False):
    combine_graph =Graph(graph1.V.union(graph2.V),graph1.lambdaPairs.union(graph2.lambdaPairs),\
        graph1.negativeNodes.union(graph2.negativeNodes),graph1.postiveLambda.union(graph2.postiveLambda))
    combine_graph.CT = graph1.CT.union(graph2.CT)
    if get_matching:
        for m1 in graph1.matching:
            if len(graph2.matching)>0:
                for m2 in graph2.matching:
                    combine_graph.matching.add(m1+m2)
            else:
                combine_graph.matching.add(m1)
    #combine
    for u in graph1.graph:
        for v in graph1.graph[u]:
            combine_graph.addEdge(u,v)
    for u in graph2.graph:
        for v in graph2.graph[u]:
            combine_graph.addEdge(u,v)
    for u in graph1.dashgraph:
        for v in graph1.dashgraph[u]:
            combine_graph.addDash(u,v)
    for u in graph2.dashgraph:
        for v in graph2.dashgraph[u]:
            combine_graph.addDash(u,v)

    #check cycle
    if combine_graph.isCyclic():
        return None

    result_graph = Graph(result_nodes,result_lambda_pairs,result_negativeNodes,result_lambda_nodes)
    result_graph.matching = combine_graph.matching

    for u in result_nodes:
        for v in result_nodes:
            if u!=v:
                isAcess = combine_graph.isAccess(u,v)
                if isAcess:
                    result_graph.addEdge(u,v)
    # print(result_graph)
    # print(combine_graph.CT)
    for u in combine_graph.dashgraph:
        for v in combine_graph.dashgraph[u]:
            if u in result_nodes and v in result_nodes:
                if combine_graph.isAccess(u,v):
                    result_graph.addEdge(u,v)
                else:
                    result_graph.addDash(u,v)
            elif u not in result_nodes and v not in result_nodes:
                if combine_graph.isAccess(u,v):
                    pass
                else:
                    return None
            elif u in result_nodes and v not in result_nodes:
                if combine_graph.isAccess(u,v):
                    pass
                else:
                    flag = False
                    for n in result_nodes:
                        if combine_graph.isAccess(n,v):
                            flag=True
                            result_graph.addDash(u,n)
                            break
                    if not flag:
                        return None
            #special case: consider later
            elif u not in result_nodes and v in result_nodes:
                if combine_graph.isAccess(u,v):
                    pass
                else:
                    return None
    if not empty_premises:
        for u in combine_graph.postiveLambda:
            if u not in combine_graph.CT:
                for v in combine_graph.negativeNodes:
                    if combine_graph.isAccess(u,v) and (not combine_graph.isAccess(u,combine_graph.negativePar[v])):
                        combine_graph.CT.add(u)
                        break
        result_graph.CT = result_nodes.intersection(combine_graph.CT)
        for u in combine_graph.postiveLambda:
            if u not in result_nodes and u not in combine_graph.CT:
                return None
    return result_graph

def add_entry(cur_entry, new_graph,empty_premiese=False, get_matching=False):
    for g in cur_entry:
        if g.isSame(new_graph,empty_premiese):
            if get_matching:
                for m in new_graph.matching:
                    g.matching.add(m)
            return
    cur_entry.add(new_graph)

def parse_sequent(cat_list, empty_premises,get_matching):
    global node_num
    node_num = 0
    terms = []
    for c in cat_list[:-1]:
        term = get_unfolding(c,NEG)
        terms.append(term)
    rhs = get_unfolding(cat_list[-1],POS)
    for term in terms:
        rhs.left_child.append(term)
        term.par = rhs
    unfolding_formulae = rhs.inorder()
    rhs.set_depth(0)
    # print(unfolding_formulae)
    f = []


    # assert [i+1 for i in range(len(unfolding_formulae))] == [e.num for e in unfolding_formulae]
    for l in range(2, len(unfolding_formulae)+1, 2):
        cur_f = []
        for s in range(1, len(unfolding_formulae)-l+2):
            e = s + l - 1
            cur_entry = set()
            result_nodes,start_path,end_path, result_lambda_nodes, result_negativeNodes, result_lambda_pairs \
                = get_result_node(unfolding_formulae, s, e)
            # print(s, e, result_nodes, start_path, end_path,result_lambda_nodes)
            #brackeing //todo
            if unfolding_formulae[s-1].prim == unfolding_formulae[e-1].prim and unfolding_formulae[s-1].sign + unfolding_formulae[e-1].sign==1:
                outer_graph = Graph(result_nodes,result_lambda_pairs,result_negativeNodes,result_lambda_nodes)
                if unfolding_formulae[s-1].sign==POS:
                    outer_graph.matching.add("("+str(e)+" "+str(s)+")")
                    pos_node = unfolding_formulae[s-1]
                    neg_node = unfolding_formulae[e-1]
                else:
                    outer_graph.matching.add("("+str(s)+" "+str(e)+")")
                    neg_node = unfolding_formulae[s-1]
                    pos_node = unfolding_formulae[e-1]
                
                
                for i,n in enumerate(start_path[:-1]):
                    if i%2==0:
                        outer_graph.addDash(n,start_path[i+1])
                    else:
                        outer_graph.addEdge(n,start_path[i+1])
                for i,n in enumerate(end_path[:-1]):
                    if i%2==0:
                        outer_graph.addDash(n,end_path[i+1])
                    else:
                        outer_graph.addEdge(n,end_path[i+1])
                outer_graph.addEdge(pos_node.num,neg_node.num)
                if not outer_graph.isCyclic():
                    if l==2:
                        cur_entry.add(combine(outer_graph,Graph(set(),set(),set(),set()),result_nodes,result_negativeNodes, result_lambda_pairs,result_lambda_nodes,get_matching))
                    elif inner_entry := f[(l-2)//2-1][s]:
                        # print("bracketing")
                        for inner_graph in inner_entry:
                            new_graph = combine(outer_graph, inner_graph,result_nodes,result_negativeNodes, result_lambda_pairs,result_lambda_nodes,get_matching)
                            if new_graph:
                                add_entry(cur_entry, new_graph,empty_premises,get_matching)
            #adjoin
            for k in range(s+1,e-1,2):
                left_entry = f[(k-s+1)//2-1][s-1]
                right_entry = f[(e-k)//2-1][k]

                if left_entry and right_entry:
                    # print("adjoin",k)
                    for graph1 in left_entry:
                        for graph2 in right_entry:
                            new_graph = combine(graph1, graph2, result_nodes,result_negativeNodes, result_lambda_pairs,result_lambda_nodes,get_matching)
                            if new_graph:
                                add_entry(cur_entry, new_graph,empty_premises,get_matching)
            if len(cur_entry)>0:
                cur_f.append(cur_entry)
            else:
                cur_f.append(None)
        # print(cur_f)
        # print()
        f.append(cur_f)

    return f
    # return unfolding_formulae

def get_match_list_of_tuple(strm):
    strm = strm.replace("("," ").replace(")"," ").split()
    result = []
    for i in range(len(strm)//2):
        result.append((int(strm[2*i]), int(strm[2*i+1])))
    return result 
def parse_txt(folder="trn", empty_premises = False, get_matching=False):
    count = 0
    idx = -1
    num_match = []
    num_lex = []
    data = {}
    max_len = 20
    gold_rank = []
    with open('./data/LCGbank.'+folder+'.str') as f:
        for line in f.readlines()[:]:
            prob_list = []
            gold_prob = 1000000
            idx += 1
            l = line.split()
            rhs = l[-1]
            if len(l)>max_len:
                continue
            print("----------------------")
            f = parse_sequent(l[:-2] + [rhs], empty_premises, get_matching)
            final_graphs =  f[-1][-1]
            answer = False
            if final_graphs:
                for g in final_graphs:
                    if sum([len(g.dashgraph[v]) for v in g.V])==0:
                        if not empty_premises and g.postiveLambda-g.CT==set():
                            if get_matching and len(g.matching)>=1:
                                if len(l) in data:
                                    data[len(l)].append(len(g.matching))
                                else:
                                    data[len(l)] = [len(g.matching)]
                                num_match.append(len(g.matching))
                                num_lex.append(len(l))
                                print(line,end='')
                                if folder=='tst':
                                    print(test_words[idx])
                                    print(test_matches[idx])
                                print(len(g.matching),"matches")
                                line = line.split()
                                line.pop(-2)
                                for curm in g.matching:
                                    
                                    curm_lst = get_match_list_of_tuple(curm)
                                    tree = proofnet_2_tree(line, curm_lst)
                                    cur_prob = 0
                                    rule = load_rule_from_tree(tree, args)
                                    for r in rule:
                                        left, right = r
                                        if left not in p_rule:
                                            cur_prob-=1000
                                        elif right not in p_rule[left]:
                                            cur_prob-=1000
                                        else:
                                            cur_prob += math.log(p_rule[left][right])*rule[r]
                                    if curm_lst==test_matches[idx]:
                                        gold_prob = cur_prob
                                    prob_list.append(cur_prob)
                            answer = True
                        elif empty_premises:
                            if get_matching:
                                print(len(g.matching))
                            answer=True
            if answer:
                print("Sequent " + str(count) + " is derivable")
                print("gold prob: ",gold_prob)
                print("rank:", np.sum(np.array(prob_list)>gold_prob+0.000000001)+1)
                gold_rank.append(np.sum(np.array(prob_list)>gold_prob+0.000000001)+1)
            else:
                print("Sequent " + str(count) + " is not derivable")
            count+=1
    plotdata = []
    a = np.array(num_match)
    b = np.array(gold_rank)
    print("1 match: ",np.sum(np.array(num_match)==1), " rank 1 ", np.sum(b==1))
    print("2 match: ",np.sum(np.array(num_match)==2), " rank 2 ", np.sum(b==2))
    print("3 match: ",np.sum(np.array(num_match)==3), " rank 3 ", np.sum(b==3))
    print("4 match: ",np.sum(np.array(num_match)==4), " rank 4 ", np.sum(b==4))
    print("5 match: ",np.sum(np.array(num_match)==5), " rank 5 ", np.sum(b==5))
    print("6-10 match: ",((6 <= a) & (a <= 10)).sum(), "6-10 rank: ",((6 <= b) & (b <= 10)).sum())
    print("11-20 match: ",((11 <= a) & (a <= 20)).sum(), "11-20 rank: ",((11 <= b) & (b <= 20)).sum())
    print("21-50 match: ",((21 <= a) & (a <= 50)).sum(), "21-50 rank: ",((21 <= b) & (b <= 50)).sum())
    print(">50 match: ",np.sum(np.array(num_match)>50), " rank > 50 ", np.sum(b>50))
    assert len(gold_rank) == len(num_match)

if __name__ == "__main__":
    folder = "tst"
    empty_premises = False
    get_matching = True
    parse_txt(folder, empty_premises, get_matching)
