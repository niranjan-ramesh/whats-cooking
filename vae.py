# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

class DeepAutoEncoder:

    def __init__(self, num_vis, num_hid_1, num_hid_2):
        super(DeepAutoEncoder, self).__init__()
        self.num_vis = num_vis
        self.num_H1 = num_hid_1
        self.num_H2 = num_hid_2

    def initialize_weights(self):
        weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2)
        with tf.name_scope('weights'):
            self.W0 = tf.Variable(name = 'w_layer0', shape = (self.num_vis, self.num_H1),
                                    initial_value=weight_initializer(shape=(self.num_vis, self.num_H1) ,dtype=tf.float64))
            self.W1 = tf.Variable(name = 'w_layer1', shape = (self.num_H1, self.num_H2),
                                    initial_value=weight_initializer(shape=(self.num_H1, self.num_H2),dtype=tf.float64) )
            self.W2 = tf.Variable(name = 'w_layer2', shape = (self.num_H2, self.num_H1),
                                    initial_value=weight_initializer(shape=(self.num_H2, self.num_H1),dtype=tf.float64))
            self.W3 = tf.Variable(name = 'w_layer3', shape = (self.num_H1, self.num_vis),
                                    initial_value=weight_initializer(shape=(self.num_H1, self.num_vis),dtype=tf.float64))


    def initialize_bias(self):

        with tf.name_scope('biases'):
            self.B0 = tf.Variable(name = 'bias0', shape = (self.num_H1),initial_value = tf.zeros(self.num_H1, dtype=tf.float64),
                                    )
            self.B1 = tf.Variable(name = 'bias1', shape = (self.num_H2),initial_value = tf.zeros(self.num_H2, dtype=tf.float64),
                                    )
            self.B2 = tf.Variable(name = 'bias2', shape = (self.num_H1),initial_value = tf.zeros(self.num_H1, dtype=tf.float64),
                                     )


    def encoder(self, X):
        '''
        Encode input X

        @param X: data to encode

        @returns Z: encoded result
        '''
        h0 =  tf.nn.bias_add(tf.matmul(X, self.W0), self.B0)
        a1 = tf.nn.sigmoid(h0)
        h1 =  tf.nn.bias_add(tf.matmul(a1, self.W1), self.B1)
        z = tf.nn.sigmoid(h1)
        return z

    def decoder(self, Z):
        '''Reconstruct the input

        @param Z: encoded input

        @returns recon_X: reconstructed input
        '''
        a2 = Z
        h2 =  tf.nn.bias_add(tf.matmul(a2, self.W2), self.B2)
        a3 = tf.nn.sigmoid(h2)
        recon_X = tf.matmul(a3, self.W3)
        return recon_X

    def validation_loss(self, x_test):
        
        ''' Computing the loss during the validation time.
    		
    	  @param x_test: test data samples
    		
    	  @return root mean squared error loss between the predicted and actual ratings
    	  '''
        
        Z = self.encoder(x_test)
        outputs = self.decoder(Z)
        mask=tf.where(tf.equal(x_test,0.0), tf.zeros_like(x_test), x_test) # identify the zero values in the test ste
        num_test_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float64) # count the number of non zero values
        bool_mask=tf.cast(mask,dtype=tf.bool) 
        outputs=tf.where(bool_mask, outputs, tf.zeros_like(outputs))
    
        MSE_loss=tf.math.divide(tf.reduce_sum(tf.square(tf.subtract(outputs,x_test))),num_test_labels)
        RMSE_loss=tf.sqrt(MSE_loss)
        
        ab_ops=tf.math.divide(tf.reduce_sum(tf.abs(tf.subtract(x_test,outputs))),num_test_labels)
            
        return RMSE_loss, ab_ops


    def train(self, X_train, X_test, epochs, batchsize, lr=0.001):

        X = tf.compat.v1.placeholder(tf.float64, [None, self.num_vis])
        X_valid = tf.compat.v1.placeholder(tf.float64, [None, self.num_vis])
        self.initialize_weights()
        self.initialize_bias()

        Z = self.encoder(X)
        recon_X = self.decoder(Z)
        
        outputs = recon_X
        mask=tf.where(tf.equal(X,0.0), tf.zeros_like(X), X) # indices of 0 values in the training set
        num_train_labels=tf.cast(tf.math.count_nonzero(mask),dtype=tf.float64) # number of non zero values in the training set
        bool_mask=tf.cast(mask,dtype=tf.bool) # boolean mask
        outputs=tf.where(bool_mask, outputs, tf.zeros_like(outputs)) # set the output values to zero if corresponding input values are zero

        loss_op=tf.math.divide(tf.reduce_sum(tf.square(tf.subtract(outputs,X))),num_train_labels)

        RMSE_loss=tf.sqrt(loss_op)
        train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_op, 
            var_list=[self.W0, self.W1, self.W2, self.W3, self.B0, self.B1, self.B2])
        test_op, ab_ops = self.validation_loss(X_valid)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            train_loss = 0.0
            for start, end in zip(range(0, len(X_train), batchsize), range(batchsize, len(X_train), batchsize)):
                batch = X_train[start:end]
                _, loss_ = sess.run([train_op, RMSE_loss], feed_dict = {
                    X: batch
                })
                train_loss += loss_
            test_loss, mean_err = sess.run([test_op, ab_ops], feed_dict = {
                X_valid: X_test
            })
            print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f, abs_error: %.3f'
                    %(epoch,(train_loss/batchsize), (test_loss), (mean_err)))

    def predict(self, X, ignore_viewed=False):

        Z = self.encoder(X)
        recon_X = self.decoder(Z)

        if ignore_viewed:
            recon_X = recon_X * (X[0] == 0)
        
        return recon_X

    def reccomend(self, user_activity, ignore_viewed):

        predicted_scores = self.predict(user_activity, ignore_viewed)


