import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
# mu,sigma = 10,35
# x= mu + sigma*np.random.random(10000)
# plt.hist(x,20,histtype='stepfilled',color='b', alpha=0.40)
# plt.show()

# trans = 10*np.random.random((3,4))
# print trans
#
# print np.transpose(trans)

# ================ sparse matrix and stuffs================
# x= np.random.RandomState(seed=123)
#
# y= x.uniform(low=3.0,high=1.0,size=(7,5))
# #print y
# y[y<2.7] = 0
# #print y
# y_car = sparse.csr_matrix(y)
# #print y_car.toarray()
# a_lil = sparse.lil_matrix((5 ,5))
# print a_lil


#==============matplotlib================


x = np.linspace(1,12 ,10)
y = x[:, np.newaxis]
# print x
# print y
# plt.interactive(True    )
# plt.plot(x,np.sin(x))
#
#
# #y=np.random.normal(size=500)
# #z=np.random.normal(size=500)
#im= y * np.sin(x) * np.cos(y)
#points = np.arange(-5, 5, 0.01)
#dx, dy = np.meshgrid(points, points)
#z = (np.sin(dx)+np.sin(dy))
#plt.imshow(im)
# # print (im.shape)
#plt.contour(x)
# # plt.scatter(y,z)
#plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# points = np.arange(-5, 5, 0.01)
# dx, dy = np.meshgrid(points, points)
# z = (np.sin(dx)+np.sin(dy))
# plt.imshow(z)
# # plt.colorbar()
# # plt.title('plot for sin(x)+sin(y)')
#
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()
fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the faces:
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
ax.imshow(faces.images[i], cmap=plt.cm.bone, interpolation='nearest')

# iris = load_iris()
# n_samples, n_features = iris.data.shape
# print (n_samples)
# print (n_features)
# print (iris.target_names )