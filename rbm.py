import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

tf.debugging.set_log_device_placement(True)


class RBM:
    def __init__(self, num_hid, num_vis):
        """Initializing the RMB architecture

        @params
        num_hid: integer
            The number of hidden activations

        num_vis: integer
            The number of visible activations

        @returns The RBM class
        """
        self.num_hid = num_hid
        self.num_vis = num_vis

        self.cur_w = np.zeros([self.num_vis, self.num_hid], np.float32)
        self.cur_bias_vis = np.zeros([self.num_vis], np.float32)
        self.cur_bias_hid = np.zeros([self.num_hid], np.float32)

        cwd = os.getcwd()
        if not os.path.exists(cwd + '/Model'):
            os.mkdir(cwd + '/Model')
        self.file_path = cwd + '/Model/rbm'

    def bernoulli_sample(self, probs):
        """Bernoulli sample to decide if the node will be sampled or not

        @params:
        probs: tensor
            The distribution of the activation of the nodes

        @returns:
        sampled: tensor
            Sampled activation of nodes
        """
        sampled = tf.nn.relu(
            tf.sign(probs - tf.random_uniform(tf.shape(probs))))
        return sampled

    def visible_to_hidden(self, v0, W, bias_hid):
        """Visible layer to hidden layer with Gibbs sampling

        @params:
        v: tensor
            The visible layer activations

        @returns
        h: tensor
            Hidden layer activation
        sampled_h: tensor
            Sampled activation from the hidden layer. Gibbs sampling is used
        """
        h = tf.nn.sigmoid(tf.matmul(v0, W) +
                          bias_hid)  # Visible layer activation
        sampled_h = self.bernoulli_sample(h)

        return h, sampled_h

    def hidden_to_visible(self, h0, W, bias_vis):
        """Hidden layer to visible layer - Reconstruction

        @params:
        h0: tensor
            The hidden layer activations

        @returns
        sampled_v: tensor
            Sampled activations from the visible layer.
        """
        _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + bias_vis)
        sampled_v = self.bernoulli_sample(_v1)
        return _v1, sampled_v

    def compute_energy(self, x):
        """Compute the free energy of the RBM

        @params:
        x: array
            Input ratings distribution

        @returns:
        energy: float
            Computed energy of the RBM
        """
        x = x.astype(np.float128)
        energy = (np.sum(np.log(1 + np.exp(np.dot(x, self.cur_w) +
                                           self.cur_bias_hid)), axis=1)
                  + np.dot(x, self.cur_bias_vis)) * -1
        return energy

    def fit(self, interactions, validation, lr, epochs, batch_size):
        """Training the RBM

        @params:
        interactions: dataframe
            User interactions
        validation: array
            Validation dataset
        lr: float
            The learning rate
        epochs: integer
            The number of epochs for training
        batch_size: integer
            The number of data points per batch
        """

        self.interactions = interactions
        bias_vis = tf.placeholder(tf.float32, [self.num_vis])
        bias_hid = tf.placeholder(tf.float32, [self.num_hid])
        W = tf.placeholder(tf.float32, [self.num_vis, self.num_hid])

        v0 = tf.placeholder(tf.float32, [None, self.num_vis])

        # Input visible to hidden layer operations

        _h0, h0 = self.visible_to_hidden(v0, W, bias_hid)

        # Reconstruction: Hidden to visible layer operations

        _v1, v1 = self.hidden_to_visible(h0, W, bias_vis)

        h1 = tf.nn.sigmoid(tf.matmul(v1, W) + bias_hid)

        # Operations to update weights and biases based on gradients
        w_pos_grad = tf.matmul(tf.transpose(v0), h0)
        w_neg_grad = tf.matmul(tf.transpose(v1), h1)

        # Contrastive Divergence to minimize the loss
        CD = (w_pos_grad - w_neg_grad) / tf.cast(tf.shape(v0)[0], tf.float32)

        update_w = W + lr * CD
        update_vb = bias_vis + lr * tf.reduce_mean(v0 - v1, 0)
        update_hb = bias_hid + lr * tf.reduce_mean(h0 - h1, 0)

        # RMSE for reconstruction
        diff = v0 - v1
        err = tf.reduce_mean(diff * diff)

        # Previous weight
        init_w = np.random.normal(loc=0, scale=0.01,
                                  size=[self.num_vis, self.num_hid])
        # Previous visible unit biases
        init_bias_vis = np.zeros([self.num_vis], np.float32)

        # Previous hidden unit biases
        init_bias_hid = np.zeros([self.num_hid], np.float32)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        train = interactions.values
        batching = {
            's': range(0, len(train), batch_size),
            'e': range(batch_size, len(train), batch_size)
        }

        train_loss = []
        valid_loss = []
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.FileWriter(train_log_dir)
        for epoch in range(epochs):
            current_loss = 0.0
            for s, e in zip(batching['s'], batching['e']):
                batch = train[s:e]
                self.cur_w = sess.run(update_w, feed_dict={
                    v0: batch, W: init_w, bias_vis: init_bias_vis, bias_hid: init_bias_hid})
                self.cur_bias_vis = sess.run(update_vb, feed_dict={
                    v0: batch, W: init_w, bias_vis: init_bias_vis, bias_hid: init_bias_hid})
                self.cur_bias_hid = sess.run(update_hb, feed_dict={
                    v0: batch, W: init_w, bias_vis: init_bias_vis, bias_hid: init_bias_hid})
                init_w = self.cur_w
                init_bias_vis = self.cur_bias_vis
                init_bias_hid = self.cur_bias_hid

            current_loss = sess.run(err, feed_dict={
                v0: train, W: self.cur_w, bias_vis: self.cur_bias_vis, bias_hid: self.cur_bias_hid
            })

            test_loss = sess.run(err, feed_dict={
                v0: validation, W: self.cur_w, bias_vis: self.cur_bias_vis, bias_hid: self.cur_bias_hid
            })

            # print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f'
            #         %(epoch,(current_loss), (test_loss)))
            train_loss.append(current_loss)
            valid_loss.append(test_loss)

        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)
        np.save(self.file_path+'/weights.npy', init_w)
        np.save(self.file_path+'/bias_vis.npy', init_bias_vis)
        np.save(self.file_path+'/bias_hid.npy', init_bias_hid)

        plt.plot(train_loss, label="Train")
        plt.plot(valid_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.savefig(self.file_path+"/error.png")

    def predict(self, user, read_file=None):
        """Predict ratings for all recipes of a user

        @params:
            user: array
                The input current ratings of a specific user

        @returns:
            predicted: array
                The predicted rating for a user
        """

        bias_vis = tf.placeholder(tf.float32, [self.num_vis])
        bias_hid = tf.placeholder(tf.float32, [self.num_hid])
        W = tf.placeholder(tf.float32, [self.num_vis, self.num_hid])

        v0 = tf.placeholder(tf.float32, [None, self.num_vis])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        inputUser = user
        if read_file is not None:
            filename = self.file_path
            self.cur_w = np.load(filename+'/weights.npy')
            self.cur_bias_vis = np.load(filename+'/bias_vis.npy')
            self.cur_bias_hid = np.load(filename+'/bias_hid.npy')

        h0, sampled_h = self.visible_to_hidden(v0, W, bias_hid)
        v1, _ = self.hidden_to_visible(h0, W, bias_vis)
        hiden_activations = sess.run(
            h0, feed_dict={v0: inputUser, W: self.cur_w,
                           bias_hid: self.cur_bias_hid})
        recon_user = sess.run(v1, feed_dict={
                              h0: hiden_activations, W: self.cur_w,
                              bias_vis: self.cur_bias_vis})

        return recon_user

    def reccomend(self, all_interactions, user_id):
        """Reccomendation scores for a user

        @params:
        all_interactions: dataframe
            The history of interactions of all users
        user_id: integer
            The id of the user to predict reccomendations for

        @returns:
        scores: dataframe
            Dataframe containing prediction scores for all recipes 
            sorted descending
        """

        user = all_interactions.loc[user_id].values
        pred_scores = self.predict([user])
        pred_scores[0] = pred_scores[0] * (user == 0)
        scores = pd.DataFrame(
            {'recipe_id': all_interactions.columns, 'scores': pred_scores})
        return scores.sort_values('scores', ascending=False)

    def write_recc_files(self):

        interactions = self.interactions
        user_ids = interactions.index
        recipe_ids = interactions.columns
        data = interactions.values
        recs = []
        preds = self.predict(data)
        pred_scores = pd.DataFrame(preds, columns=recipe_ids, index=user_ids).T
        for user_id in user_ids:
            user_pred = pred_scores[[user_id]].sort_values(
                user_id, ascending=False)
            recipes_rec = user_pred.index[:10]
            recs.append(recipes_rec)
        reccomendations = pd.DataFrame(recs, columns=range(1, 11))
        reccomendations['user_id'] = user_ids
        cwd = os.getcwd()
        reccomendations.to_csv(
            cwd + '/Data/reccomendations_rbm.csv', index=False)
