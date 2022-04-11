import os

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


class DeepAutoEncoder:

    def __init__(self, num_vis, num_hid_1, num_hid_2):
        super(DeepAutoEncoder, self).__init__()
        self.num_vis = num_vis
        self.num_H1 = num_hid_1
        self.num_H2 = num_hid_2

        cwd = os.getcwd()
        if not os.path.exists(cwd + '/Model'):
            os.mkdir(cwd + '/Model')
        self.file_path = cwd + '/Model/vae'

    def initialize_weights(self):
        weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.2)
        with tf.name_scope('weights'):
            self.W0 = tf.Variable(name='w_layer0',
                                  shape=(self.num_vis, self.num_H1),
                                  initial_value=weight_initializer(
                                      shape=(self.num_vis, self.num_H1),
                                      dtype=tf.float64))
            self.W1 = tf.Variable(name='w_layer1',
                                  shape=(self.num_H1, self.num_H2),
                                  initial_value=weight_initializer(
                                      shape=(self.num_H1, self.num_H2),
                                      dtype=tf.float64))
            self.W2 = tf.Variable(name='w_layer2',
                                  shape=(self.num_H2, self.num_H1),
                                  initial_value=weight_initializer(
                                      shape=(self.num_H2, self.num_H1),
                                      dtype=tf.float64))
            self.W3 = tf.Variable(name='w_layer3',
                                  shape=(self.num_H1, self.num_vis),
                                  initial_value=weight_initializer(
                                      shape=(self.num_H1, self.num_vis),
                                      dtype=tf.float64))

    def initialize_bias(self):

        with tf.name_scope('biases'):
            self.B0 = tf.Variable(name='bias0', shape=(self.num_H1),
                                  initial_value=tf.zeros(
                                      self.num_H1, dtype=tf.float64),
                                  )
            self.B1 = tf.Variable(name='bias1', shape=(self.num_H2),
                                  initial_value=tf.zeros(
                                      self.num_H2, dtype=tf.float64),
                                  )
            self.B2 = tf.Variable(name='bias2', shape=(self.num_H1),
                                  initial_value=tf.zeros(
                                      self.num_H1, dtype=tf.float64),
                                  )

    def encoder(self, X):
        '''
        Encode input X

        @param X: data to encode

        @returns Z: encoded result
        '''
        h0 = tf.nn.bias_add(tf.matmul(X, self.W0), self.B0)
        a1 = tf.nn.sigmoid(h0)
        h1 = tf.nn.bias_add(tf.matmul(a1, self.W1), self.B1)
        z = tf.nn.sigmoid(h1)
        return z

    def decoder(self, Z):
        '''Reconstruct the input

        @param Z: encoded input

        @returns recon_X: reconstructed input
        '''
        a2 = Z
        h2 = tf.nn.bias_add(tf.matmul(a2, self.W2), self.B2)
        a3 = tf.nn.sigmoid(h2)
        recon_X = tf.matmul(a3, self.W3)
        return recon_X

    def validation_loss(self, x_test):
        ''' Computing the loss during the validation time.

          @param x_test: test data samples

          @return Reconsturcion loss for validation
          '''

        Z = self.encoder(x_test)
        recon_x = self.decoder(Z)
        mask = tf.where(tf.equal(x_test, 0.0), tf.zeros_like(x_test), x_test)
        num_test_labels = tf.cast(tf.count_nonzero(mask), dtype=tf.float64)
        bool_mask = tf.cast(mask, dtype=tf.bool)
        recon_x = tf.where(bool_mask, recon_x, tf.zeros_like(recon_x))

        ab_ops = tf.math.divide(tf.reduce_sum(
            tf.abs(tf.subtract(x_test, recon_x))), num_test_labels)

        return ab_ops

    def fit(self, interactions, X_test, epochs, batchsize, lr=0.001):

        self.interactions = interactions

        X = tf.compat.v1.placeholder(tf.float64, [None, self.num_vis])
        X_valid = tf.compat.v1.placeholder(tf.float64, [None, self.num_vis])
        self.initialize_weights()
        self.initialize_bias()

        Z = self.encoder(X)
        recon_X = self.decoder(Z)

        mask = tf.where(tf.equal(X, 0.0), tf.zeros_like(X), X)
        num_train_labels = tf.cast(
            tf.math.count_nonzero(mask), dtype=tf.float64)
        bool_mask = tf.cast(mask, dtype=tf.bool)
        outputs = tf.where(bool_mask, recon_X, tf.zeros_like(recon_X))

        loss_op = tf.math.divide(tf.reduce_sum(
            tf.square(tf.subtract(outputs, X))), num_train_labels)

        RMSE_loss = tf.sqrt(loss_op)
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            loss_op, var_list=[self.W0, self.W1, self.W2, self.W3,
                               self.B0, self.B1, self.B2])
        test_op = self.validation_loss(X_valid)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        data = interactions.values
        train_errors = []
        test_errors = []
        for epoch in range(epochs):
            train_loss = 0.0
            for start, end in zip(range(0, len(data), batchsize), range(batchsize, len(data), batchsize)):
                batch = data[start:end]
                _, loss_ = sess.run([train_op, loss_op], feed_dict={
                    X: batch
                })
                # train_loss += loss_
            train_loss = sess.run(RMSE_loss, feed_dict={
                X: batch
            })
            test_loss = sess.run(test_op, feed_dict={
                X_valid: X_test
            })

            # p, r = self.evaluate(data)

            # print('Precision: ', p)
            # print('Recall: ', r)
            train_errors.append(train_loss)
            test_errors.append(test_loss)
            # print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f' %(epoch,(train_loss), test_loss))

        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)

        plt.plot(test_errors, label="Train")
        plt.plot(train_errors, label="Validation")
        plt.xlabel('Loss')
        plt.ylabel('Epoch')

        plt.savefig(self.file_path + '/error.png')

    def predict(self, user):
        """Predict ratings for all recipes of a user

        @params:
            user: array
                The input current ratings of a specific user

        @returns:
            predicted: array
                The predicted rating for a user
        """

        U = tf.compat.v1.placeholder(tf.float64, [None, self.num_vis])

        Z = self.encoder(U)
        recon_X = self.decoder(Z)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        pred = sess.run(recon_X, feed_dict={U: user})

        return pred

    def evaluate(self, X):
        """Evaluation metrics for reccomendation"""

        scores = self.predict(X)

        preds = pd.DataFrame(scores,
                             index=self.interactions.index,
                             columns=self.interactions.columns).T

        return self.recall_at_k(X, scores), self.precision_at_k(X, scores)

    def precision_at_k(self, actual, predictions, k=10, relevance=3):
        pt = 0.0
        pr = 0.0
        total_act = 0.0
        for a, p in zip(actual, predictions):
            ratings = sorted(zip(a, p), key=lambda x: x[1])
            if(len(ratings) > k):
                ratings = ratings[:k]
            for pair in ratings:
                if(pair[0] > relevance and pair[1] > relevance):
                    pr += 1
                elif(p[1] > relevance):
                    pt += 1
        if(total_act == 0.0):
            return 0.0
        return pr/pr

    def recall_at_k(self, actual, predictions, k=10, relevance=3):
        pr = 0.0
        ar = 0.0
        for a, p in zip(actual, predictions):
            ratings = sorted(zip(a, p), key=lambda x: x[1])
            if(len(ratings) > k):
                ratings = ratings[:k]
            for pair in ratings:
                if(pair[1] > relevance):
                    pr += 1
                if(pair[0] > relevance):
                    ar += 1
        if(ar == 0.0):
            return 0.0
        return pr/ar

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
            cwd + '/Data/reccomendations_vae.csv', index=False)
