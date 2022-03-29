from audioop import bias
import dataclasses
import torch
import torch.nn as nn

import torch.optim as optim

from torch.nn import functional as func

class RBM:
    """
    Constructing a Restircted Boltzmann Machine

    Required Args:

    num_vis: Size of the visible layer
    num_hid: Size of the hidden layer
    """

    def __init__(self, num_vis, num_hid) -> None:
        super(RBM, self).__init__()
        self.W = torch.randn(num_vis, num_hid, dtype=torch.float64)
        self.bias_vis = torch.randn(num_vis, dtype=torch.float64)
        self.bias_hid = torch.randn(num_hid, dtype=torch.float64)
        self.criterion = nn.MSELoss()

    def visible_to_hidden(self, X):
        a = torch.mm(X, self.W)
        h = torch.sigmoid(a + self.bias_hid.expand_as(a))
        return h
    
    def hidden_to_visible(self, H):
        x = torch.mm(H, self.W.t())
        v = torch.sigmoid(x + self.bias_vis.expand_as(x))
        return v


    def forward(self, X):
        """Compute the hidden and reconstructed inputs
        """
        h0 = self.visible_to_hidden(X)
        # Gibb's sample
        sampled_h = func.relu(torch.sign(h0 - torch.rand(h0.size())))
        return sampled_h

    def recon(self, h):
        v = self.hidden_to_visible(h)
        sampled_v = func.relu(torch.sign(v - torch.rand(v.size())))
        h1 = self.visible_to_hidden(sampled_v)
        return v, h1

    def recon_loss(self, X_test):
        X_test = torch.tensor(X_test)
        sam_h_test = self.forward(X_test)
        v, h1 = self.recon(sam_h_test)

        mask=torch.where(X_test == 0.0, torch.zeros_like(X_test), X_test) # indices of 0 values in the testing set
        num_test_labels=torch.count_nonzero(mask).type(torch.DoubleTensor) # number of non zero values in the testing set
        bool_mask=mask.type(torch.BoolTensor) # boolean mask
        outputs=torch.where(bool_mask, v, torch.zeros_like(v)) # set the output values to zero if corresponding input values are zero

        MSE_loss=torch.divide(torch.sum(torch.square(torch.sub(outputs,X_test))),num_test_labels)
        RSME_loss = torch.sqrt(MSE_loss)
        
        return RSME_loss

    def prec_at_10(self, X_test):
        X_test = torch.tensor(X_test)
        sample_hidden = self.forward(X_test)
        v, h1 = self.recon(sample_hidden)


    def train(self, X_train, X_test, epochs, lr, batchsize):

        for epoch in range(epochs):
            train_loss = 0.0
            test_loss = 0.0
            mean_err = 0.0
            for start, end in zip(range(0, len(X_train), batchsize), range(batchsize, len(X_train), batchsize)):
                batch = X_train[start:end]
                X = torch.tensor(batch)

                sampled_h = self.forward(X)
                v, h1 = self.recon(sampled_h)

                mask=torch.where(X == 0.0, torch.zeros_like(X), X) # indices of 0 values in the training set
                num_train_labels=torch.count_nonzero(mask).type(torch.DoubleTensor) # number of non zero values in the training set
                bool_mask=mask.type(torch.BoolTensor) # boolean mask
                outputs=torch.where(bool_mask, v, torch.zeros_like(v)) # set the output values to zero if corresponding input values are zero

                MSE_loss=torch.divide(torch.sum(torch.square(torch.sub(outputs,X))),num_train_labels)
                RSME_loss = torch.sqrt(MSE_loss)

                w_pos_grad = torch.mm(X.t(), sampled_h)
                w_neg_grad = torch.mm(outputs.t(), h1)

                CD = (w_pos_grad - w_neg_grad) / X.shape[0]

                self.W += (lr * CD)
                self.bias_vis += (lr * torch.mean(X - outputs, 0))
                self.bias_hid += (lr * torch.mean(sampled_h - h1, 0))

                train_loss += RSME_loss
            
            
            print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f, abs_error: %.3f'
                    %(epoch,(train_loss/batchsize), (test_loss), (mean_err)))

            



