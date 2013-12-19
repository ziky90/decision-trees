import monkdata as m
import dtree as dt
import drawtree as draw
import matplotlib.pyplot as plt
import random, operator

#Entorpy

#calling the predefined function that calculates the entropy for all the three datasets
#assignment 1
print dt.entropy(m.monk1)
print dt.entropy(m.monk2)
print dt.entropy(m.monk3)
print '\n'

##############################################################################
#Information Gain

#cycles for calling average gains for all the three datasets and for every attribute
#assignment 2
for atr in m.attributes:
    gain = dt.averageGain(m.monk1, atr)
    print gain
    
print '\n'    
for atr in m.attributes:
    print dt.averageGain(m.monk2, atr)

print '\n'    
for atr in m.attributes:
    print dt.averageGain(m.monk3, atr)
print '\n' 

 
#############################################################################
#Building decision trees

#function that selects the attribute with the highest expected information gain
def bestAttribute(dataset, attributes):
    result = 0
    best = attributes[0]
    for a in attributes:
        value = dt.averageGain(dataset, a)
        if value > result:
            result = value
            best = a
    return best


#splitting the data
a = bestAttribute(m.monk1, m.attributes)
data = []
for v in a.values:
    data.append(dt.select(m.monk1, a, v))

#calculating the average information gain for the next level
for d in data:
    for a in m.attributes:
        print dt.averageGain(d, a)
    print '\n'
print '\n' 

#comparison with the tree from the predefined function
tree = dt.buildTree(m.monk1, m.attributes, 2)
#draw.drawTree(tree)


#building the trees for all the monks datasets
#assignment 3
tree1 = dt.buildTree(m.monk1, m.attributes)
print dt.check(tree1, m.monk1)
print dt.check(tree1, m.monk1test)
#draw.drawTree(tree)
print '\n'

tree2 = dt.buildTree(m.monk2, m.attributes)
print dt.check(tree2, m.monk2)
print dt.check(tree2, m.monk2test)
#draw.drawTree(tree)
print '\n'

tree3 = dt.buildTree(m.monk3, m.attributes)
print dt.check(tree3, m.monk3)
print dt.check(tree3, m.monk3test)
#draw.drawTree(tree)
print '\n'


###############################################################################
#Prunning
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

monk1train, monk1val = partition(m.monk1, 0.6)


def prun(tree, val):
    candidates = {}
    pruns = dt.allPruned(tree)
    for p in pruns:
        performance = dt.check(p, val)
        candidates[p] = performance
    return candidates

candidates = prun(tree1, monk1val)
bestTree = max(candidates.iteritems(), key=operator.itemgetter(1))[0]
#draw.drawTree(bestTree)

#Assignment 4
fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
classificationError1 = []
classificationError3 = []
for f in fraction:
    monk1train, monk1val = partition(m.monk1, f)
    tree1 = dt.buildTree(monk1train, m.attributes)
    candidates1 = prun(tree1, monk1val)
    classificationError1.append(dt.check(max(candidates1.iteritems(), key=operator.itemgetter(1))[0], m.monk1test))
    
    monk3train, monk3val = partition(m.monk3, f)
    tree3 = dt.buildTree(monk3train, m.attributes)
    candidates3 = prun(tree3, monk3val)
    classificationError3.append(dt.check(max(candidates3.iteritems(), key=operator.itemgetter(1))[0], m.monk3test))

    
plt.plot(fraction, classificationError1, 'r', label='monk1')
plt.plot(fraction, classificationError3, 'b', label='monk3')
plt.title('performance with respect to fraction')
plt.xlabel('fraction')
plt.ylabel('performance')
plt.legend()
plt.show()
