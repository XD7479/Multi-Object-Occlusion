import numpy as np
import scipy
if tuple(map(int, scipy.__version__.split('.'))) < (1, 0, 0):
    from scipy.misc import logsumexp
else:
    from scipy.special import logsumexp
import math
import time
from datetime import date
import os
import pickle

def normalize_features(features):
	'''features: n by d matrix'''
	assert(len(features.shape)==2)
	norma=np.sqrt(np.sum(features ** 2, axis=1).reshape(-1, 1))+1e-6
	return features/norma

class vMFMM:
	def __init__(self, cls_num, init_method = 'random', tmp_dir = '/tmp/'):
		self.cls_num = cls_num
		self.init_method = init_method

		if not os.path.exists(tmp_dir):
			os.makedirs(tmp_dir)

		self.tmp_file = os.path.join(tmp_dir, str(date.today())+'.pickle')


	def fit(self, features, kappa, max_it=300, tol = 5e-5, normalized=False, verbose=True):
		self.features = features
		if not normalized:
			self.features = normalize_features(features)

		self.n, self.d = self.features.shape
		self.kappa = kappa

		self.pi = np.random.random(self.cls_num)
		self.pi /= np.sum(self.pi)
		if self.init_method =='random':
			self.mu = np.random.random((self.cls_num, self.d))
			# self.mu = np.array([[1,0],[0,1]])
			self.mu = normalize_features(self.mu)
		elif self.init_method =='k++':
			#print('start k++')
			centers = []
			centers_i = []

			if self.n > 50000:
				rdn_index = np.random.choice(self.n, size=(50000,), replace=False)
			else:
				rdn_index = np.array(range(self.n), dtype=int)

			cos_dis = 1-np.dot(self.features[rdn_index], self.features[rdn_index].T)

			#print('finish cos_dis')
			centers_i.append(np.random.choice(rdn_index))
			centers.append(self.features[centers_i[0]])
			for i in range(self.cls_num-1):
				#if i%10==0:
			#		print('k++ center {0}'.format(i))

				cdisidx = [np.where(rdn_index==cci)[0][0] for cci in centers_i]
				prob = np.min(cos_dis[:,cdisidx], axis=1)**2
				prob /= np.sum(prob)
				centers_i.append(np.random.choice(rdn_index, p=prob))
				centers.append(self.features[centers_i[-1]])

			self.mu = np.array(centers)
			del(cos_dis)
			#print('finish k++')

		self.mllk_rec = []
		for itt in range(max_it):
			_st = time.time()
			self.e_step()
			self.m_step()
			_et = time.time()
			#if verbose and itt%1==0:
				#print("iter {0}: {1}, time: {2}".format(itt, self.mllk, (_et-_st)/60))

			if itt%20==0:
				with open(self.tmp_file, 'wb') as fh:
					pickle.dump(self.mu, fh)

				bins = 4
				per_bin = self.cls_num//bins+1
				for bb in range(bins):
					with open(self.tmp_file.replace('.pickle','_p{}.pickle'.format(bb)), 'wb') as fh:
						pickle.dump(self.p[:,bb*per_bin:(bb+1)*per_bin], fh)

			self.mllk_rec.append(self.mllk)
			if len(self.mllk_rec)>1 and self.mllk - self.mllk_rec[-2] < tol:
				print("early stop at iter {0}, llk {1}".format(itt, self.mllk))
				break


	def fit_soft(self, features, p, mu, pi, kappa, max_it=300, tol = 1e-6, normalized=False, verbose=True):
		self.features = features
		if not normalized:
			self.features = normalize_features(features)

		self.p = p
		self.mu = mu
		self.pi = pi
		self.kappa = kappa

		self.n, self.d = self.features.shape

		for itt in range(max_it):
			self.e_step()
			self.m_step()
			#if verbose and itt%20==0:
				#print("iter {0}: {1}".format(itt, self.mllk))

			self.mllk_rec.append(self.mllk)
			if len(self.mllk_rec)>1 and self.mllk - self.mllk_rec[-2] < tol:
				print("early stop at iter {0}, llk {1}".format(itt, self.mllk))
				break


	def e_step(self):
		# update p
		logP = np.dot(self.features, self.mu.T)*self.kappa + np.log(self.pi).reshape(1,-1)  # n by k
		logP_norm = logP - logsumexp(logP, axis=1).reshape(-1,1)
		self.p = np.exp(logP_norm)
		self.mllk = np.mean(logsumexp(logP, axis=1))


	def m_step(self):
		# update pi and mu
		self.pi = np.sum(self.p, axis=0)/self.n

		# fast version, requires more memory
		self.mu = np.dot(self.p.T, self.features)/np.sum(self.p, axis=0).reshape(-1,1)

#         d_cut = 52
#         bnum = int(math.ceil(self.d/d_cut))

#         for dd_i in range(bnum):
#             dd_start = dd_i*d_cut
#             dd_end = min((dd_i+1)*d_cut, self.d)
#             self.mu[:,dd_start:dd_end] = np.sum(np.tile(self.features.reshape(self.n,1,self.d)[:,:,dd_start:dd_end],(1,self.cls_num,1))*self.p.reshape(self.n,self.cls_num,1),axis=0)/np.sum(self.p, axis=0).reshape(-1,1)

		# for cc in range(self.cls_num):
		#     self.mu[cc] = np.sum(self.p[:,cc].reshape(-1,1) * self.features, axis=0)/np.sum(self.p[:,cc])

		# r = np.mean(np.sqrt(np.sum(self.mu**2, axis=1))*self.pi)
		# r = np.mean(np.sqrt(np.sum(self.mu**2, axis=1))/(self.n*self.pi))
		# self.kappa2 = (r*self.d-r**3)/(1-r**2)

		self.mu = normalize_features(self.mu)





