
import os
import time
import pickle
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from truncated_normal import truncated_normal as tn
from scipy.stats import ttest_ind


def plot_labels_legend(x1, x2, Y):
    for i in np.unique(Y):
        plt.plot(x1[Y == i], x2[Y == i], '.',
                 label=r'%s ($n$ = %s)'%(i, np.sum(Y==i)))
    plt.legend()


adata = sc.read_10x_mtx('data/filtered_gene_bc_matrices/hg19/', var_names='gene_symbols', cache=True)
X = adata.X
features = np.array(adata.var['gene_ids'].index)
tsne = np.array(pd.read_csv(os.path.join('./pbmc_seurat_tsne.txt'), delimiter=' '))

np.random.seed(0)

picklefile = 'pbmc_tntest_tutorial.pickle'
if os.path.isfile(picklefile):
    results = pickle.load(open(picklefile, 'rb'))
else:
    results = {}

start = time.time()
for i, (c1, c2) in enumerate(itertools.combinations(np.unique(labels2), 2)):
    if (c1, c2) in results:
        continue
    p_t = ttest_ind(X1[labels1 == c1].todense(), X1[labels1 == c2].todense())[1]
    p_t[np.isnan(p_t)] = 1
    y = np.array(X2[labels2 == c1].todense())
    z = np.array(X2[labels2 == c2].todense())
    a = np.array(svm.coef_[i].todense()).reshape(-1)
    b = svm.intercept_[i]
    p_tn, likelihood = tn.tn_test(y, z, a=a, b=b,
                                  learning_rate=1.,
                                  eps=1e-2,
                                  verbose=True,
                                  return_likelihood=True,
                                  num_iters=100000,
                                  num_cores=64)
    results[(c1, c2)] = (p_t, p_tn)
    print('c1: %5s\tc2: %5s\ttime elapsed: %.2fs' % (c1, c2, time.time() - start))
    pickle.dump(results, open(picklefile, 'wb'))