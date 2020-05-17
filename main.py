import numpy as np
import pandas as pd
import scipy as scp
import matplotlib.pyplot as plt
import os
import pickle
import tensorflow as tf

def df_conv(x,y,dtyp):
	if dtyp=='pandas':
		return (x.values, y.values)
	elif dtyp=='numpy':
		return (x, y)
	else:
		print('Invalid dtyp variable')
		sys.exit()

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def wvar(sd,shp):
	return tf.Variable(tf.truncated_normal(shp,stddev=sd))

def bvar(val,shp):
	return tf.Variable(tf.constant(val,shape=shp))

class aeknn(object):

	def __init__(self, typ, x, y, aeshape=[32,64,128], dense=30, dtyp='pandas', cnn=False, imshp=(28,28), kernel=3, strides=[1,2,2,1], padding='SAME', weight_sd_init=.1, bias_init=.1):
		"""Arguments:
			typ: String specification on whether model is classifier or regressor ('class','reg').
			x: Independent variable data with same structure of data that will be trained/predicted on. Used as part of setting autoencoder structure.
			y: Dependent variable data with same structure of data that will be trained/predicted on. Used as part of setting autoencoder structure.
			aeshape: List of integers defining depth/shape of encoder/decoder.
			dense: Nodes in dense encoded layer.
			dtyp: Defines datatable type that will be used moving forward ('pandas' or 'tf')
			cnn: Whether cnn (default not)
			imshp: If cnn, dimension to orient input
			kernel: If cnn, kernel shape
			strides: If cnn, stride size
			padding: If cnn, padding method
			weight_sd_init: SD for weight matrix intialization
			bias_init: Bias magnitude intitialization
		"""

		#Set class attributes
		self.typ=typ
		self.aeshape=aeshape
		self.dense=dense
		self.dtyp=dtyp
		self.cnn=cnn
		self.imshp=imshp
		self.kernel=kernel
		self.strides=strides
		self.padding=padding

		#Define input
		self.x=tf.placeholder(tf.float32,[None,x.shape[1]])
		#print(x.shape)
		#x_image=tf.reshape(self.x,[-1, self.imshp[0], self.imshp[1], 1])
		if cnn:
			x=tf.reshape(self.x,[-1, self.imshp[0], self.imshp[1], 1])
		else:
			x=tf.reshape(self.x,[-1, x.shape[1]])

		#Define output
		self.y=tf.placeholder(tf.int8,[None])

		#Define ratio multiplier of output constrained loss
		self.ratio=tf.placeholder(tf.float32)

		#Define batch size
		batch=tf.shape(self.x)[0]

		#Build encoder based on specified shape
		enc=[]
		if cnn:
			f_lag=1
		else:
			f_lag=int(x.shape[1])
		h_lag=x
		#For loop iterates through layers
		for filt in self.aeshape:
			#Weights
			if cnn:
				enc.append([wvar(weight_sd_init, [self.kernel, self.kernel, f_lag, filt])])
			else:
				enc.append([wvar(weight_sd_init, [f_lag, filt])])
			#Bias
			enc[-1].append(bvar(bias_init, [filt]))
			if cnn:
				enc[-1].append(tf.nn.relu(tf.nn.conv2d(h_lag, enc[-1][0], strides=self.strides, padding=self.padding)+enc[-1][1]))
			else:
				enc[-1].append(tf.nn.relu(tf.linalg.matmul(h_lag, enc[-1][0])+enc[-1][1]))
			#Set prior layer filter reference to use for next layer input channels
			f_lag=filt
			h_lag=enc[-1][2]
		#Encode dense
		if cnn:
			dim1,dim2=h_lag.get_shape().as_list()[1:3]
		else:
			dim1,dim2=(1,1)
		flat=tf.reshape(h_lag,[-1,dim1*dim2*f_lag])
		enc.append([wvar(weight_sd_init, [dim1*dim2*f_lag,self.dense])])

		enc[-1].append(bvar(bias_init, [self.dense]))
		enc[-1].append(tf.matmul(flat,enc[-1][0])+enc[-1][1])

		#Store dense encoded layer
		self.encoded_x=enc[-1][2]

		#Decode dense
		dec=[]
		dec.append([wvar(weight_sd_init, [self.dense,dim1*dim2*f_lag])])
		dec[-1].append(bvar(bias_init, [dim1*dim2*f_lag]))
		dec[-1].append(tf.matmul(enc[-1][2],dec[-1][0])+dec[-1][1])
		#Build decoder mirroring encoder configuration
		if cnn:
			h_lag=tf.reshape(dec[-1][2], [-1, dim1, dim2, f_lag])
		else:
			h_lag=tf.reshape(dec[-1][2], [-1, f_lag])

		for filt in reversed(self.aeshape[:-1]):
			#Weights
			if cnn:
				dec.append([wvar(weight_sd_init, [self.kernel, self.kernel, filt, f_lag])])
			else:
				dec.append([wvar(weight_sd_init, [f_lag, filt])])
			#Bias
			dec[-1].append(bvar(bias_init, [filt]))
			if cnn:
				#Output shape
				ref=tf.shape([x[2] for x in enc if x[1].get_shape().as_list()[0]==filt][0])
				out_shape=tf.stack([batch,ref[1],ref[2],ref[3]])
				dec[-1].append(tf.nn.relu(tf.nn.conv2d_transpose(h_lag, dec[-1][0],output_shape=out_shape, strides=self.strides, padding=self.padding)+dec[-1][1]))
			else:
				dec[-1].append(tf.nn.relu(tf.linalg.matmul(h_lag, dec[-1][0])+dec[-1][1]))
			#Set prior layer filter reference to use for next layer input channels
			f_lag=filt
			h_lag=dec[-1][2]
		if cnn:
			dec.append([wvar(weight_sd_init, [self.kernel,self.kernel,1,f_lag])])
			dec[-1].append(bvar(bias_init, [1]))
			out_shape=tf.stack([batch, self.imshp[0], self.imshp[1], 1])
			dec[-1].append(tf.nn.sigmoid(tf.nn.conv2d_transpose(h_lag, dec[-1][0],output_shape=out_shape, strides=self.strides, padding=self.padding)+dec[-1][1]))
		else:
			dec.append([wvar(weight_sd_init, [f_lag, 784])])
			dec[-1].append(bvar(bias_init, [784]))
			#out_shape=tf.stack([batch, 784])
			dec[-1].append(tf.nn.sigmoid(tf.linalg.matmul(h_lag, dec[-1][0])+dec[-1][1]))

		#Handle reconstructed output
		self.reconstructed_x=dec[-1][2]
		if cnn:
			reconx_flat=tf.reshape(dec[-1][2],[-1, self.imshp[0]*self.imshp[1]])
		else:
			reconx_flat=tf.reshape(dec[-1][2],[-1, x.shape[1]])
		recon_err=tf.negative(tf.reduce_sum(self.x*tf.log(reconx_flat+1e-10) + (1-self.x)*tf.log(1-reconx_flat+1e-10)))

		#Classification v. Regression Error
		if self.typ=='class':
			#NCA penalty
			dx=tf.subtract(self.encoded_x[:, None], self.encoded_x[None])
			masks=tf.equal(self.y[:, None], self.y[None])

			sftmax_zero_diag=tf.matrix_set_diag(tf.reduce_sum(tf.exp(-tf.square(dx)),2),tf.zeros([batch]))
			sftmax=sftmax_zero_diag/tf.reduce_sum(sftmax_zero_diag, 1)

			nca=tf.reduce_sum(tf.where(masks, sftmax, tf.zeros([batch,batch])))

			fm=tf.cast(batch, tf.float32)

			self.nca=tf.negative(tf.div(nca,fm))
		elif self.typ=='reg':
			# MSE Penalty
			tmp_enc=np.array(self.encoded_x).astype(np.float)
			tmp_y=np.array(self.y).flatten()
			x_square = np.diag(np.dot(tmp_enc, tmp_enc.T))
			ed_matrix = (np.ones((len(tmp_enc), 1)) * x_square.T) - 2 * (np.dot(tmp_enc, tmp_enc.T))
			label_index_array = np.argpartition(ed_matrix, k+1, axis=1)[:, 1:k+1]
			preds = trny[label_index_array]
			preds = np.mean(preds.T, axis=0)
			#need to finish
		self.recon_err=tf.div(recon_err,fm)
		self.loss=tf.multiply(self.ratio, self.nca)+tf.multiply(tf.subtract(1., self.ratio), self.recon_err)

	def train(self, x, y, save, learn_rate=.001, batch=100, epoch=100, ratio=.0, load=None):
		"""Arguments:
		x: Independent variable dataframe to be trained on
		y: Dependent variable dataframe to be trained on
		save: Model checkpoint saving name/directory
		learn_rate: Neural net learning rate
		batch: Batch size
		epoch: Epoch size
		ratio: Weighting ratio for constraint to loss function (user must keep in mind magnitude differential)
		load: Checkpoint location if starting with prior training
		"""

		optimizer=tf.train.AdamOptimizer(learn_rate).minimize(self.loss)

		#saver=tf.train.Saver(tf.global_variables())
		saver=tf.train.Saver()
		sess=tf.Session()
		sess.run(tf.global_variables_initializer())

		if load!=None:
			saver.restore(sess, load)

		for epoch_i in range(epoch):
			for batch_i in range(len(y)//batch):
				batch_x, batch_y=next_batch(batch, x, y)
				sess.run(optimizer, feed_dict={self.x:batch_x, self.y:batch_y, self.ratio:ratio})
			errs=[[],[],[]]
			for batch_i in range(len(y)//batch):
				batch_x, batch_y=next_batch(batch, x, y)
				err=(sess.run([self.recon_err, self.nca, self.loss], feed_dict={self.x:batch_x, self.y:batch_y, self.ratio:ratio}))
				for i in range(3):
					errs[i].append(err[i])
			print('Epoch'+str(epoch_i), np.mean(errs[0]), np.mean(errs[1]), np.mean(errs[2]))

		saver.save(sess, save)


	def predict(self, typ, trnx, trny, tstx, k, load):
		"""Arguments:
		typ: Prediction method (can be knn or particle filter ('KNN', 'PFIL'))
		trnx: Reference indpendent variable data for neighbors
		trny: Rereference dependent variable data for neighbors
		tstx: Indpendent variable data to predict on
		k: If typ is 'KNN', number of neighbors for prediction
		load: Trained autoencoder to load
		"""

		saver=tf.train.Saver(tf.global_variables())
		sess=tf.Session()
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, load)

		trn_enc=np.array(sess.run(self.encoded_x, feed_dict={self.x: trnx, self.y: trny, self.ratio:0.})).astype(np.float)
		tst_enc=np.array(sess.run(self.encoded_x, feed_dict={self.x: tstx, self.y: np.array(len(trnx)*[0]), self.ratio:0.})).astype(np.float)

		trny=np.array(trny).flatten()
		x_square = np.diag(np.dot(trn_enc, trn_enc.T))
		ed_matrix = (np.ones((len(tst_enc), 1)) * x_square.T) - 2 * (np.dot(tst_enc, trn_enc.T))
		label_index_array = np.argpartition(ed_matrix, k, axis=1)[:, :k]
		preds = trny[label_index_array]
		for i, p in enumerate(preds):
			if len(np.unique(p)) == len(p):
				preds[i][-1] = preds[i][0]
		preds = scp.stats.mode(preds.T).mode[0]

		return preds
