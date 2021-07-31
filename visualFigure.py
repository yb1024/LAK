import scipy.io as scio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
#
# data0=scio.loadmat('FigureData2/data0.mat')['data']
# label0=scio.loadmat('FigureData2/label0.mat')['data']
#
# Visualdata0 = TSNE(learning_rate=500).fit_transform(data0)
#
# plt.figure(1)
# plt.scatter(Visualdata0[:, 0], Visualdata0[:, 1], c=np.squeeze(label0))
# plt.show()
#
# data90=scio.loadmat('FigureData2/data90.mat')['data']
# label90=scio.loadmat('FigureData2/label90.mat')['data']
#
# Visualdata90 = TSNE(learning_rate=500).fit_transform(data90)
#
# plt.figure(2)
# plt.scatter(Visualdata90[:, 0], Visualdata90[:, 1], c=np.squeeze(label90))
# plt.show()

scio.savemat('FigureData2/Visualdata0.mat', {'data': Visualdata0})
scio.savemat('FigureData2/Visualdata90.mat', {'data': Visualdata90})


'''优化前可视化'''
Visualdata0=scio.loadmat('FigureData2/Visualdata0.mat')['data']
label0=scio.loadmat('FigureData2/label0.mat')['data']
plt.figure(1)
plt.scatter(Visualdata0[:, 0], Visualdata0[:, 1], c=np.squeeze(label0))
plt.show()

'''优化后可视化'''
Visualdata0=scio.loadmat('FigureData2/Visualdata0.mat')['data']
label0=scio.loadmat('FigureData2/label0.mat')['data']
plt.figure(2)
plt.scatter(Visualdata90[:, 0], Visualdata90[:, 1], c=np.squeeze(label90))
plt.show()

