import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from data import *
import math
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Probabilistic Lambek Categorial Sequent")

    parser.add_argument("--par_depth", type=int, default="2", help="depth of parant")
    parser.add_argument("--lambda_is_par", action='store_true', help="consider lambda node as parant node or not")
    
    args = parser.parse_args()
    
    return args

def load_rule_from_tree(t, args):
    #print(t)
    rule = {("",t.name):1}
    queue = [t]
    while queue!=[]:
        cur = queue.pop(0)
        left = cur.prim
        con = cur.con
        p = cur.par
        par_left = args.par_depth - 1
        while p!=None and par_left>0:
            left = p.prim + con + left
            if p.prim!="L" and not args.lambda_is_par:
                par_left -= 1
            elif args.lambda_is_par:
                par_left -= 1
            con = p.con
            p = p.par
        r = (left,",".join([c.name for c in cur.child]))
        if r in rule:
            rule[r] += 1
        else:
            rule[r] = 1
        for c in cur.child:
            queue.append(c)
    return rule
if __name__=='__main__':
    args = parse_arguments()
    print(args)
    train_trees, train_words, train_matches = load_trees('trn')
    test_trees, test_words, test_matches = load_trees('tst')
    dev_trees, dev_words, dev_matches = load_trees('dev')
    all_rule = {}
    print(len(train_trees),len(dev_trees), len(test_trees))
    count = 0
    for t in train_trees[:]:
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
    curpus_level_prob = 0
    seen_sen = 0
    total_yield = 0
    for i, t in enumerate(test_trees[:]):
        try:
            cur_prob = 0
            rule = load_rule_from_tree(t, args)
            # print(rule)
            for r in rule:
                left, right = r
                cur_prob += math.log(p_rule[left][right])*rule[r]
            curpus_level_prob+=cur_prob
            seen_sen+=1
            total_yield += t.count_cat()
        except:
            pass
    print("seen sentence:",seen_sen)
    print("NLL:",curpus_level_prob/total_yield)
    
