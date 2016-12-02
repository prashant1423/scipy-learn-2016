import matplotlib.pyplot as plt
import numpy as np
mu,sigma = 10,35
x= mu + sigma*np.random.random(10000)
plt.hist(x,20,histtype='stepfilled',color='b', alpha=0.40)
plt.show()