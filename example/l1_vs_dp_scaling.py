import numpy as np
import matplotlib.pyplot as plt
import pdb

scl = 0.01

if __name__ == '__main__':
	d = np.arange(0, 21)
	x = np.floor(np.pow(1.6, d)) # up to 12k
	l1 = np.zeros(x.shape)
	dp = np.zeros(x.shape)
	l1_std = np.zeros(x.shape)
	dp_std = np.zeros(x.shape)

	for i in range(x.shape[0]):
		d = int(x[i])
		f = max(1000, d*2)
		a = np.random.randn(f,d)
		b = np.random.randn(f,d)
		# apply a random matrix projection, as would be done in a NN.
		wa = np.random.randn(d,d)*scl
		wb = np.random.randn(d,d)*scl
		a = a@wa
		b = b@wb
		l1[i] = np.mean(np.sum(np.abs(a-b), axis=1))
		dp[i] = np.mean(np.sum(a*b, axis=1))
		l1_std[i] = np.std(np.sum(np.abs(a-b), axis=1))
		dp_std[i] = np.std(np.sum(a*b, axis=1))

	fig,axs = plt.subplots(2,2)

	axs[0,0].plot(x,l1, label='l1')
	axs[0,0].set_title("L1, mean")
	axs[0,0].get_xaxis().set_ticks([])

	axs[0,1].plot(x,dp, label='DP')
	axs[0,1].set_title("DP, mean")
	axs[0,1].get_xaxis().set_ticks([])

	axs[1,0].plot(x,l1_std, label='l1')
	axs[1,0].set_title("L1, std")
	axs[1,0].set(xlabel='dimension')

	axs[1,1].plot(x,dp_std, label='DP')
	axs[1,1].set_title("DP, std")
	axs[1,1].set(xlabel='dimension')

	plt.show()
