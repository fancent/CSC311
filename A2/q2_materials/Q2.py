import numpy as np
import matplotlib.pyplot as plt
from utils import load_train, load_valid
from run_knn import run_knn

trainData = load_train()
validData = load_valid()

kRange = [1,3,5,7,9]
results = []
for k in kRange:
    temp = run_knn(k, trainData[0],trainData[1],validData[0])
    results.append(temp)
    
def classificationRate(validSet, trainResult):
    return np.sum(validSet == trainResult)/len(validSet)

classificationRateResults = [classificationRate(validData[1], i) for i in results]

fig, graph = plt.subplots()
graph.plot(kRange, classificationRateResults, 'x')
graph.plot(kRange, classificationRateResults)
graph.set(xlabel='k value', ylabel='classification rate',
       title='classification rate as a function of k')
graph.grid()
fig.savefig("q2_1.png")
plt.show()
