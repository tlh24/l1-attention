import numpy as np
import matplotlib.pyplot as plt
import pdb
import math

scl = 0.01

if __name__ == '__main__':
	d = np.arange(0, 21)
	x = np.floor(np.pow(1.6, d)) # up to 12k
	l1 = np.zeros(x.shape)
	dp = np.zeros(x.shape)
	l2 = np.zeros(x.shape)
	l1_std = np.zeros(x.shape)
	dp_std = np.zeros(x.shape)
	l2_std = np.zeros(x.shape)

	for i in range(x.shape[0]):
		d = int(x[i])
		f = max(1000, d*2) # number of samples
		a = np.random.randn(f,d)
		b = np.random.randn(f,d)
		# apply a random matrix projection, as would be done in a NN.
		kaiming = math.sqrt(2 / d) # kaiming initialization
		wa = np.random.randn(d,d)*kaiming
		wb = np.random.randn(d,d)*kaiming
		a = a@wa
		b = b@wb
		l1_d = np.sum(np.abs(a-b), axis=1) - \
			0.7071*(np.sum(np.abs(a), axis=1) + np.sum(np.abs(b), axis=1))
		dp_d = np.sum(a*b, axis=1)
		l2_d = np.sqrt(np.sum((a-b)**2, axis=1))
		l1[i] = np.mean(l1_d)
		dp[i] = np.mean(dp_d)
		l2[i] = np.mean(l2_d)
		l1_std[i] = np.std(l1_d)
		dp_std[i] = np.std(dp_d)
		l2_std[i] = np.std(l2_d)

	fig,axs = plt.subplots(2,3, figsize=(10,5))

	axs[0,0].plot(x,l1, label='l1')
	axs[0,0].set_title("L1, mean, Kaiming init")
	axs[0,0].get_xaxis().set_ticks([])

	axs[0,1].plot(x,dp, label='DP')
	axs[0,1].set_title("DP, mean")
	axs[0,1].get_xaxis().set_ticks([])

	axs[0,2].plot(x,l2, label='L2')
	axs[0,2].set_title("L2, mean")
	axs[0,2].get_xaxis().set_ticks([])

	axs[1,0].plot(x,l1_std, label='l1')
	axs[1,0].set_title("L1, std")
	axs[1,0].set(xlabel='dimension')

	axs[1,1].plot(x,dp_std, label='DP')
	axs[1,1].set_title("DP, std")
	axs[1,1].set(xlabel='dimension')

	axs[1,2].plot(x,l2_std, label='L2')
	axs[1,2].set_title("L2, std")
	axs[1,2].set(xlabel='dimension')

	plt.show()
