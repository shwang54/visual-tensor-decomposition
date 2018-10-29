# --------------------------------------------------------
# Tensorize, Factorize and Regularize: Robust Visual Relationship Learning
# Written by Seong Jae Hwang, Zirui Tao
#
# Adopted from Scene Graph Generation by Iterative Message Passing
# https://github.com/danfeiX/scene-graph-TF-release
# Licensed under The MIT License
# Written by Danfei Xu
# --------------------------------------------------------
"""
Train a scene graph generation network
"""

import tensorflow as tf
import numpy as np
import os

from fast_rcnn.config import cfg
from networks.factory import get_network
from networks import losses
from roi_data_layer.data_runner import DataRunnerMP
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer

class Trainer(object):

	def __init__(self, sess, net_name, imdb, roidb, output_dir, tf_log,  pretrained_model=None, sigma = cfg.SIGMA):
		"""Initialize the SolverWrapper."""
		self.net_name = net_name
		self.imdb = imdb
		self.roidb = roidb
		self.output_dir = output_dir
		self.tf_log = tf_log
		self.pretrained_model = pretrained_model
		self.bbox_means = np.zeros((self.imdb.num_classes, 4))
		self.bbox_stds = np.ones((self.imdb.num_classes, 4))

		# sigma passed in
		self.sigma = sigma

		if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
			print('Loaded precomputer bbox target distribution from %s' % \
				  cfg.TRAIN.BBOX_TARGET_NORMALIZATION_FILE)
			bbox_dist = np.load(cfg.TRAIN.BBOX_TARGET_NORMALIZATION_FILE).item()
			self.bbox_means = bbox_dist['means']
			self.bbox_stds = bbox_dist['stds']

		print 'Trainer initialization done'


	def snapshot(self, sess, iter):
		"""Take a snapshot of the network after unnormalizing the learned
		bounding-box regression weights. This enables easy use at test-time.
		"""
		net = self.net

		if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers and cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
			# save original values
			with tf.variable_scope('bbox_pred', reuse=True):
				weights = tf.get_variable("weights")
				biases = tf.get_variable("biases")

			orig_0 = weights.eval()
			orig_1 = biases.eval()
			# scale and shift with bbox reg unnormalization; then save snapshot
			weights_shape = weights.get_shape().as_list()
			sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds.ravel(), (weights_shape[0],1))))
			sess.run(biases.assign(orig_1 * self.bbox_stds.ravel() + self.bbox_means.ravel()))

		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		filename = os.path.join(self.output_dir, 'weights_%i.ckpt' % iter)

		self.saver.save(sess, filename)
		print 'Wrote snapshot to: {:s}'.format(filename)

		if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers and cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
			# restore net to original state
			sess.run(weights.assign(orig_0))
			sess.run(biases.assign(orig_1))


	def get_data_runner(self, sess, data_layer):

		input_pls = {
			'ims': tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]),
			'rois': tf.placeholder(dtype=tf.float32, shape=[None, 5]),
			'rel_rois': tf.placeholder(dtype=tf.float32, shape=[None, 5]),
			'labels': tf.placeholder(dtype=tf.int32, shape=[None]),
			'relations': tf.placeholder(dtype=tf.int32, shape=[None, 2]),
			'predicates': tf.placeholder(dtype=tf.int32, shape=[None]),
			'bbox_targets': tf.placeholder(dtype=tf.float32, shape=[None, 4 * self.imdb.num_classes]),
			'bbox_inside_weights': tf.placeholder(dtype=tf.float32, shape=[None, 4 * self.imdb.num_classes]),
			'num_roi': tf.placeholder(dtype=tf.int32, shape=[]),  # number of rois per batch
			'num_rel': tf.placeholder(dtype=tf.int32, shape=[]),  # number of relationships per batch
			'rel_mask_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
			'rel_segment_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
			'rel_pair_mask_inds': tf.placeholder(dtype=tf.int32, shape=[None,2]),
			'rel_pair_segment_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
			'Xr': tf.placeholder(dtype=tf.float32, shape=[151, 151, 51]),
			'sigma': tf.placeholder(dtype=tf.float32, shape=[]),
			'mask': tf.placeholder(dtype=tf.float32, shape=[None,51])

		}
		def data_generator():
			while True:
				yield data_layer.next_batch()

		def task_generator():
			while True:
				yield data_layer._get_next_minibatch_inds()

		task_func = data_layer._get_next_minibatch
		data_runner = DataRunnerMP(task_func, task_generator, input_pls, capacity=24)

		return data_runner


	def train_model(self, sess, max_iters):
		"""Network training loop."""
		data_layer = RoIDataLayer(self.imdb, self.bbox_means, self.bbox_stds)

		# a multi-process data runner
		data_runner = self.get_data_runner(sess, data_layer)

		inputs= data_runner.get_inputs()

		inputs['num_classes'] = self.imdb.num_classes
		inputs['num_predicates'] = self.imdb.num_predicates
		inputs['n_iter'] = cfg.TRAIN.INFERENCE_ITER

		self.net = get_network(self.net_name)(inputs)
		self.net.setup()

		# # loading from the graph
		pretrained_Xu = 'checkpoints/Xu_2.4_pt-mul/weights_149999.ckpt'
		print ('Loading model weights from {:s}'.format(pretrained_Xu))
		cfg.TRAIN.SNAPSHOT_FREQ = max_iters/10
		print 'saving model every {:d} iter'.format(cfg.TRAIN.SNAPSHOT_FREQ)
		self.saver = tf.train.Saver(max_to_keep = None)


		# get network-defined losses

		ops = self.net.losses()
		# ops_rel = self.net._ours_()
		# print ops_rel
		# multitask loss
		loss_list = [ops[k] for k in ops if k.startswith('loss')]
		ops['loss_total'] = losses.total_loss_and_summaries(loss_list, 'total_loss')



		# optimizer
		lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
		momentum = cfg.TRAIN.MOMENTUM



		optimizer = tf.train.MomentumOptimizer(lr, momentum).minimize(ops['loss_total'])
		ops['train']= optimizer
		ops_summary = dict(ops)
		#merge summaries
		ops_summary['summary'] = tf.summary.merge_all()
		train_writer =  tf.summary.FileWriter(self.tf_log, sess.graph)

		# self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

		# init_loss_op = tf.initialize_variables(loss_list)
		sess.run(tf.global_variables_initializer())

		self.saver.restore(sess, pretrained_Xu)

		data_runner.start_processes(sess, n_processes=3, sigma = self.sigma)
		# intialize variables

		if self.pretrained_model is not None:
			print ('Loading pretrained model '
				   'weights from {:s}').format(self.pretrained_model)
			if self.pretrained_model.endswith('.npy'):
				self.net.load(self.pretrained_model, sess, load_fc=True)
			elif self.pretrained_model.endswith('.ckpt'):
				self.saver.restore(sess, self.pretrained_model)
			else:
				print('Unsupported pretrained weights format')
				raise

		last_snapshot_iter = -1
		timer = Timer()
		iter_timer = Timer()

		# Training loop
		import random

		for iter in range(max_iters):
			# learning rate
			iter_timer.tic()
			if (iter+1) % cfg.TRAIN.STEPSIZE == 0:
				sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))

			# Make one SGD update
			feed_dict = data_runner.get_feed_batch()
			feed_dict[self.net.keep_prob] = 0.5
			# timer.tic()
			# ops_value = sess.run(ops_rel, feed_dict=feed_dict)
			# print "--------------cls_score_from_ours--------------"
			# print ops_value['cls_score']
			# timer.toc()

			timer.tic()

			if (iter + 1) % cfg.TRAIN.SUMMARY_FREQ == 0:
				ops_value = sess.run(ops_summary, feed_dict=feed_dict)
				train_writer.add_summary(ops_value['summary'], iter)
			else:
				ops_value = sess.run(ops, feed_dict=feed_dict)

			timer.toc()
			# print "-------------cls_score_1_1------------"
			# print ops_value['cls_score']



			# ops_2['train'] = optimizer
			# np.random.seed(0)
			# ops_value = sess.run(ops_2, feed_dict=feed_dict)
			# ops_2.pop('train', None)

			stats = 'iter: %d / %d, lr: %f' % (iter+1, max_iters, lr.eval())
			for k in ops_value:
				if k.startswith('loss'):
					stats += ', %s: %4f' % (k, ops_value[k])
			print(stats)

			# print "-------------cls_score_0------------"
			# print ops_value['cls_score_0']
			# print "-------------cls_score_1_2------------"
			# print ops_value['cls_score']
			# # print "-------------------------"
			# #
			#
			# #
			# # print "-------------predicates ----------"
			# # print ops_value['predicates']
			# # print ops_value['predicates'].shape
			# #
			# # print "-------------cls_score------------"
			# # print ops_value['cls_score']
			# # print ops_value['cls_score'].shape
			# #
			#
			# print "-------------cls_pred------------"
			# print ops_value['cls_pred']
			# print ops_value['cls_pred'].shape
			#
			# # print "---------rel_rois----------------"
			# # print ops_value['rel_rois'][:10]
			# # print ops_value['rel_rois'].shape
			# #
			#
			#
			# print "-----------labels--------------"
			# print ops_value['labels']
			# print ops_value['labels'].shape
			#
			# print "-----------Xr--------------"
			# # print ops_value['Xr']
			# print ops_value['Xr'].shape
			#
			# print "-----------indices--------------"
			# print ops_value['indices']
			# print ops_value['indices'].shape
			#
			# print "-----------Relations--------------"
			# print ops_value['relations']
			# print ops_value['relations'].shape
			#
			# print "--------Rel_Score-----------------"
			# print ops_value['rel_score']
			# print ops_value['rel_score'].shape
			# print "--------MASK-----------------"
			# print ops_value['mask']
			# print ops_value['mask'].shape
			# print "--------RES-----------------"
			# print ops_value['res']
			# print ops_value['res'].shape


			iter_timer.toc()

			if (iter+1) % (10 * cfg.TRAIN.DISPLAY_FREQ) == 0:
				print 'speed: {:.3f}s / iter'.format(timer.average_time)
				print 'iter speed: {:.3f}s / iter'.format(iter_timer.average_time)

			if (iter+1) % cfg.TRAIN.SNAPSHOT_FREQ == 0:
				last_snapshot_iter = iter
				self.snapshot(sess, iter)

		if last_snapshot_iter != iter:
			self.snapshot(sess, iter)


def train_net(network_name, imdb, roidb, output_dir, tf_log, pretrained_model=None, max_iters=200000, sigma = cfg.SIGMA):
	config = tf.ConfigProto()
	config.allow_soft_placement=True
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		tf.set_random_seed(cfg.RNG_SEED)
		trainer = Trainer(sess, network_name, imdb, roidb, output_dir, tf_log, pretrained_model=pretrained_model, sigma=sigma)
		trainer.train_model(sess, max_iters)
