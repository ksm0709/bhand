import time
import csv
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

n_sne = 10000

data = open('./normal_base.tsv')

tsvreader = csv.reader(data, delimiter='\t')

tsvdata=[]
c_=[]
numline=-1
for line in tsvreader:
    numline += 1
    if numline == 0 :
        continue

    temp =[]
    for x in line[0:-1]:
       temp.append(float(x)) 

    tsvdata.append(temp)
    c_.append(temp[8]) 

fitdata = np.array(tsvdata)
c=np.array(c_)

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=1000)

tsne_results = tsne.fit_transform( fitdata )

plt.scatter(tsne_results[:,0], tsne_results[:,1],c=c)
plt.show()
