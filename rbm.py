# Import PyTorch library
import torch
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from utils import Util

# Create the Restricted Boltzmann Machine architecture
class RBM:
    def __init__(self, alpha, H, num_vis):

        self.alpha = alpha
        self.num_hid = H
        self.num_vis = num_vis # might face an error here, call preprocess if you do
        self.errors = []
        self.energy_train = []
        self.energy_valid = []

    def sample_h(self, x):
        """
        Sample the hidden units
        :param x: the dataset
        """

        # Probability h is activated given that the value v is sigmoid(Wx + a)
        # torch.mm make the product of 2 tensors
        # W.t() take the transpose because W is used for the p_v_given_h
        wx = torch.mm(x, self.W.t())

        # Expand the mini-batch
        activation = wx + self.h_bias.expand_as(wx)

        # Calculate the probability p_h_given_v
        p_h_given_v = torch.sigmoid(activation)

        # Construct a Bernoulli RBM to predict whether an user loves the movie or not (0 or 1)
        # This corresponds to whether the n_hid is activated or not activated
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """
        Sample the visible units
        :param y: the dataset
        """

        # Probability v is activated given that the value h is sigmoid(Wx + a)
        wy = torch.mm(y, self.W)

        # Expand the mini-batch
        activation = wy + self.v_bias.expand_as(wy)

        # Calculate the probability p_v_given_h
        p_v_given_h = torch.sigmoid(activation)

        # Construct a Bernoulli RBM to predict whether an user loves the movie or not (0 or 1)
        # This corresponds to whether the n_vis is activated or not activated
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def training(self, train, valid, user, epochs, batchsize, free_energy, verbose, filename):
        '''
        Function where RBM training takes place
        '''
        vb = tf.placeholder(tf.float32, [self.num_vis]) # Number of unique books
        hb = tf.placeholder(tf.float32, [self.num_hid]) # Number of features were going to learn
        W = tf.placeholder(tf.float32, [self.num_vis, self.num_hid])  # Weight Matrix
        v0 = tf.placeholder(tf.float32, [None, self.num_vis])

        print("Phase 1: Input Processing")
        _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
        # Gibb's Sampling
        h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
        print("Phase 2: Reconstruction")
        _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
        v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
        h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

        print("Creating the gradients")
        w_pos_grad = tf.matmul(tf.transpose(v0), h0)
        w_neg_grad = tf.matmul(tf.transpose(v1), h1)

        # Calculate the Contrastive Divergence to maximize
        CD = (w_pos_grad - w_neg_grad) / tf.cast(tf.shape(v0)[0], tf.float32)

        # Create methods to update the weights and biases
        update_w = W + self.alpha * CD
        update_vb = vb + self.alpha * tf.reduce_mean(v0 - v1, 0)
        update_hb = hb + self.alpha * tf.reduce_mean(h0 - h1, 0)

        # Set the error function, here we use Mean Absolute Error Function
        err = v0 - v1
        err_sum = tf.reduce_mean(err * err)

        # Initialize our Variables with Zeroes using Numpy Library
        # Current weight
        cur_w = np.zeros([self.num_vis, self.num_hid], np.float32)
        # Current visible unit biases
        cur_vb = np.zeros([self.num_vis], np.float32)

        # Current hidden unit biases
        cur_hb = np.zeros([self.num_hid], np.float32)

        # Previous weight
        prv_w = np.random.normal(loc=0, scale=0.01,
                                size=[self.num_vis, self.num_hid])
        # Previous visible unit biases
        prv_vb = np.zeros([self.num_vis], np.float32)

        # Previous hidden unit biases
        prv_hb = np.zeros([self.num_hid], np.float32)

        print("Running the session")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        print("Training RBM with {0} epochs and batch size: {1}".format(epochs, batchsize))
        print("Starting the training process")
        util = Util()
        for i in range(epochs):
            for start, end in zip(range(0, len(train), batchsize), range(batchsize, len(train), batchsize)):
                batch = train[start:end]
                cur_w = sess.run(update_w, feed_dict={
                                 v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_vb = sess.run(update_vb, feed_dict={
                                  v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_hb = sess.run(update_hb, feed_dict={
                                  v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                prv_w = cur_w
                prv_vb = cur_vb
                prv_hb = cur_hb

            if valid:
                etrain = np.mean(util.free_energy(train, cur_w, cur_vb, cur_hb))
                self.energy_train.append(etrain)
                evalid = np.mean(util.free_energy(valid, cur_w, cur_vb, cur_hb))
                self.energy_valid.append(evalid)
            self.errors.append(sess.run(err_sum, feed_dict={
                          v0: train, W: cur_w, vb: cur_vb, hb: cur_hb}))
            if verbose:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))
            elif i % 10 == 9:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))
        if not os.path.exists('rbm_models'):
            os.mkdir('rbm_models')
        filename = 'rbm_models/'+filename
        if not os.path.exists(filename):
            os.mkdir(filename)
        np.save(filename+'/w.npy', prv_w)
        np.save(filename+'/vb.npy', prv_vb)
        np.save(filename+'/hb.npy',prv_hb)
        
        if free_energy:
            print("Exporting free energy plot")
            # self.export_free_energy_plot(filename)
        print("Exporting errors vs epochs plot")
        # self.export_errors_plot(filename)
        inputUser = [train[user]]
        # Feeding in the User and Reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
        rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
        return rec, prv_w, prv_vb, prv_hb
