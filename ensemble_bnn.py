# Rohitash Chandra, 2017 c.rohitash@gmail.conm

# !/usr/bin/python

# built using: https://github.com/rohitash-chandra/VanillaFNN-Python


# Sigmoid units used in hidden and output layer. gradient descent and stocastic gradient descent functions implemented with momentum


# Multi-task learning for modular neural networks.


import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math

import scipy
from scipy import stats


# An example of a class
class Network:
    def __init__(self, Topo, Train, Test, MaxTime, MinPer):

        self.Top = Topo  # NN topology [input, hidden, output]
        self.Max = MaxTime  # max epocs
        self.TrainData = Train
        self.TestData = Test
        self.NumSamples = Train.shape[0]

        self.lrate = 0  # will be updated later with BP call

        self.momenRate = 0

        self.minPerf = MinPer
        # initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
        np.random.seed()
        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.BestB1 = self.B1
        self.BestW1 = self.W1
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer
        self.BestB2 = self.B2
        self.BestW2 = self.W2
        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def printNet(self):
        print self.Top
        print self.W1

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer#
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired):

        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = np.zeros(self.Top[2])
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

    # update weights and bias
        layer = 1  # hidden to output
        for x in xrange(0, self.Top[layer]):
            for y in xrange(0, self.Top[layer + 1]):
                self.W2[x, y] += self.lrate * out_delta[y] * self.hidout[x]
        for y in xrange(0, self.Top[layer + 1]):
            self.B2[y] += -1 * self.lrate * out_delta[y]

        layer = 0  # Input to Hidden

        for x in xrange(0, self.Top[layer]):
            for y in xrange(0, self.Top[layer + 1]):
                self.W1[x, y] += self.lrate * hid_delta[y] * Input[x]
        for y in xrange(0, self.Top[layer + 1]):
            self.B1[y] += -1 * self.lrate * hid_delta[y]


    def saveKnowledge(self):
        self.BestW1 = self.W1
        self.BestW2 = self.W2
        self.BestB1 = self.B1
        self.BestB2 = self.B2


    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]

        self.total_weightsbias = w_layer1size + w_layer2size + self.Top[1] + self.Top[2]


    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w

    def evaluate_proposal(self,   w):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = self.TrainData.shape[0]
        Input = np.zeros((1, self.Top[0]))  # temp hold input
        fx = np.zeros(size)



        for pat in xrange(0, size):
            Input[:] =  self.TrainData[pat, 0:self.Top[0]]
            self.ForwardPass(Input)
            fx[pat] = self.out
        return fx

    def test_proposal(self,  w):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = self.TestData.shape[0]
        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        sse = 0

        for pat in xrange(0, size):
            Input[:] = self.TestData[pat, 0:self.Top[0]]
            Desired[:] = self.TestData[pat, self.Top[0]:]
            self.ForwardPass(Input)
            fx[pat] = self.out
            sse = sse + self.sampleEr(Desired)

        rmse = np.sqrt(sse / size)


        return [fx,rmse]




# -------

# --------------------------------------------------------------------------------------------------------





class BayesNN:  # Multi-Task leaning using Stocastic GD

    def __init__(self, mtaskNet, traindata, testdata, samples, minPerf, learnRate, num_subtasks):
        # trainData and testData could also be different datasets. this example considers one dataset

        self.traindata = traindata
        self.testdata = testdata
        self.samples = samples
        self.minCriteria = minPerf
        self.subtasks = num_subtasks  # number of network modules
        # need to define network toplogies for the different tasks.

        self.mtaskNet = mtaskNet

        self.learnRate = learnRate
        self.trainTolerance = 0.20  # [eg 0.15 output would be seen as 0] [ 0.81 would be seen as 1]
        self.testTolerance = 0.49

    def rmse(self, predictions, targets):

        return np.sqrt(((predictions - targets) ** 2).mean())

    def net_size(self, netw):
        return ((netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2])

    def likelihood_func(self, neuralnet,  y,  w, tausq):
        #print y, ' ..y'

        fx = neuralnet.evaluate_proposal(w)
        #print fx, ' fx'
        #print y, ' y'
        rmse = self.rmse(fx, y)
        #print rmse, '  .. rmse  '
        #print fx[0:5],  ' ...fx'
        #print y[0:5], ' ... y'
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]



    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq, topo):
        h = topo[1]  # number hidden neurons
        d = topo[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def taskdata(self, data, taskfeatures, output):
        # group taskdata from main data source.
        # note that the grouping is done in accending order fin terms of features.
        # the way the data is grouped as tasks can change for different applications.
        # there is some motivation to keep the features with highest contribution as first  feature space for module 1.
        datacols = data.shape[1]
        featuregroup = data[:, 0:taskfeatures]
        return np.concatenate((featuregroup[:, range(0, taskfeatures)], data[:, range(datacols - output, datacols)]),
                              axis=1)


    def mcmc_sampler(self):

    # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples


        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)


        Netlist = [None] * 10  # create list of Network objects ( just max size of 10 for now )

        samplelist = [None] * samples  # create list of Network objects ( just max size of 10 for now )

        rmsetrain = np.zeros(self.subtasks)
        rmsetest  = np.zeros(self.subtasks)
        trainfx = np.random.randn(self.subtasks, trainsize)
        testfx = np.random.randn(self.subtasks, testsize)

        netsize = np.zeros(self.subtasks, dtype=np.int)


        depthSearch = 5  # declare


        for n in xrange(0, self.subtasks):
            module = self.mtaskNet[n]
            trdata = self.taskdata(self.traindata, module[0], module[2])  # make the partitions for task data
            testdata = self.taskdata(self.testdata, module[0], module[2])
            Netlist[n] = Network(self.mtaskNet[n], trdata, testdata, depthSearch, self.minCriteria)


        trdata = self.taskdata(self.traindata, module[0], module[2])  # make the partitions for task data
        testdata = self.taskdata(self.testdata, module[0], module[2])





        for n in xrange(0, self.subtasks):
            netw = Netlist[n].Top
            netsize[n] =  self.net_size(netw)  # num of weights and bias
            print netsize[n]


        y_test = testdata[:, netw[0]]  #grab the actual predictions from dataset
        y_train = trdata[:, netw[0]]

        w_pos = np.zeros((samples, self.subtasks, netsize[self.subtasks-1]))  # 3D np array

        posfx_train = np.zeros((samples, self.subtasks, trainsize))
        posfx_test = np.zeros((samples, self.subtasks, testsize))

        posrmse_train = np.zeros((samples, self.subtasks))
        posrmse_test = np.zeros((samples, self.subtasks))

        pos_tau = np.zeros(samples)

        print posrmse_test
        print posfx_test
        print pos_tau, ' pos_tau'




        w = np.random.randn( self.subtasks, netsize[self.subtasks-1])

        w_pro =  np.random.randn( self.subtasks, netsize[self.subtasks-1])


        step_w = 0.05;  # defines how much variation you need in changes to w
        step_eta = 0.01

        print 'evaluate Initial w'




        pred_train = Netlist[0].evaluate_proposal(w[0,:netsize[0]])  # we only take prior calculation for first ensemble, since we have one tau value for all the ensembles.

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0



        likelihood = np.zeros(self.subtasks)
        likelihood_pro = np.zeros(self.subtasks)

        prior_likelihood = np.zeros(self.subtasks)

        prior_pro = np.zeros(self.subtasks)

        for n in xrange(0, self.subtasks):
            prior_likelihood[n] = self.prior_likelihood(sigma_squared, nu_1, nu_2, w[n,:netsize[0]], tau_pro,  Netlist[n].Top)  # takes care of the gradients

        print  prior_likelihood, ' .. prior'

        mh_prob = np.zeros(self.subtasks)

        for s in xrange(0, self.subtasks):
           [likelihood[s],  posfx_train[0, s,:],  posrmse_train[0, s]] = self.likelihood_func(Netlist[s], y_train, w[s,:netsize[s]], tau_pro)

        naccept = 0


        print likelihood, ' ..likelihood'

        print 'being sampling'

        for i in xrange(1, samples-1):    # ---------------------------------------

            for s in xrange(0, self.subtasks):
                w_pro[s, :netsize[s]] = w[s, :netsize[s]] + np.random.normal(0, step_w, netsize[s])



            eta_pro  = eta  + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            for s in xrange(0, self.subtasks):
                [likelihood_pro[s],  trainfx[s, :], rmsetrain[s]] = self.likelihood_func(Netlist[s], y_train, w_pro[s, :netsize[s]], tau_pro)

                [testfx[s, :],rmsetest[s]] = Netlist[s].test_proposal(w[s,:netsize[s]])



            for n in xrange(0, self.subtasks):
                prior_pro[n] = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_pro[s, :netsize[s]], tau_pro, Netlist[n].Top)


            diff = likelihood_pro  - likelihood
            diff_prior = prior_pro - prior_likelihood


            for s in xrange(0, self.subtasks):
                #print s, '   ... s'
                mh_prob[s] = min(1, math.exp(diff[s] + diff_prior[s]))



                u = random.uniform(0, 1)


                if u < mh_prob[s]:
                    naccept += 1
                    print i, ' is accepted sample'
                    likelihood[s] = likelihood_pro[s]
                    w[s,:netsize[s]] = w_pro[s,:netsize[s]]  # _w_proposal
                    #x_ = x_pro



                    eta = eta_pro

                    prior_likelihood[s] = prior_pro[s]

                    #print rmsetrain[s]

                    print likelihood_pro[s], prior_pro[s], rmsetrain[s], rmsetest[s], s,  '   for s'  # takes care of the gradients

                    print likelihood_pro, prior_pro, rmsetrain, rmsetest, '   for all'  # takes care of the gradients

                    w_pos[i+1, s, :netsize[s]] = w_pro[s, :netsize[s]]


                    posfx_train[i+1, s, :] = trainfx[s, :]
                    posfx_test[i+1, s, :] = testfx[s, :]

                    posrmse_train[i+1,s] = rmsetrain[s]
                    posrmse_test[i+1,s] = rmsetest[s]

                    pos_tau[i+1] = tau_pro





                else:

                    w_pos[i + 1, s, :netsize[s]] = w_pos[i, s, :netsize[s]]

                    posfx_train[i+1, s, :] = posfx_train[i, s, :]
                    posfx_test[i+1, s, :] = posfx_test[i, s, :]

                    posrmse_train[i+1,s] = posrmse_train[i,s]
                    posrmse_test[i+1,s] = posrmse_test[i,s]

                    pos_tau[i + 1] =  pos_tau[i]






        print naccept, ' num accepted'
        accept_ratio = naccept / (samples * 1.0) * 100


        return (w_pos, posfx_train, posfx_test, posrmse_train, posrmse_test, pos_tau, x_train, x_test,  y_test, y_train, accept_ratio)




# ------------------------------------------------------------------------------------------------------



def main():
    #np.random.seed()
    fileout1 = open('test.txt', 'w')
    fileout2 = open('Results_bnn_dts.txt', 'w')

    moduledecomp = [1 ,1,  1]  # decide what will be number of features for each group of taskdata correpond to module

    for problem in range(2, 3):

        hidden = 4
        input = 4  #
        output = 1
        learnRate = 0.2
        mRate = 0.01
        num_samples = 2000

        if problem == 1:
            traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")  #
        if problem == 2:
            traindata = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt")  #
        if problem == 3:
            traindata = np.loadtxt("Data_OneStepAhead/Mackey/train7.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Mackey/test7.txt")  #
        if problem == 4:
            traindata = np.loadtxt("Data_OneStepAhead/Lorenz/train7.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Lorenz/test7.txt")  #
        if problem == 5:
            traindata = np.loadtxt("Data_OneStepAhead/Rossler/train7.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Rossler/test7.txt")  #
        if problem == 6:
            traindata = np.loadtxt("Data_OneStepAhead/Henon/train7.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Henon/test7.txt")  #
        if problem == 7:
            traindata = np.loadtxt("Data_OneStepAhead/ACFinance/train7.txt")
            testdata = np.loadtxt("Data_OneStepAhead/ACFinance/test7.txt")  #




        min_perf = 0.0000001  # stop when RMSE reches this point

        numModules = 3  # first decide number of  modules (or ensembles for comparison)

        baseNet = [input, hidden, output]

        inputfeatures = baseNet[0]  # total num inputfeatures for the prob

        mtaskNet = np.array([baseNet, baseNet, baseNet])

        for i in xrange(1, numModules):
            mtaskNet[i - 1][0] = moduledecomp[i - 1] * inputfeatures
            mtaskNet[i][1] += (i * 1)  # in this example, we have fixed numner  output neurons. input for each task is termined by feature group size.
            # we adapt the number of hidden neurons for each task.
        print mtaskNet  # print network topology of all the modules that make the respective tasks. Note in this example, the tasks aredifferent network topologies given by hiddent number of hidden layers.



        bayesnn = BayesNN(mtaskNet, traindata, testdata, num_samples, min_perf, learnRate, numModules)


        #bayesnn.sampler()

        [w_pos, posfx_train, posfx_test, posrmse_train, posrmse_test, pos_tau, x_train, x_test, y_test, y_train, accept_ratio] = bayesnn.mcmc_sampler()

        fx_train = posfx_train[:,0,:]
        fx_test = posfx_test[:,0,:]

        rmse_train = posrmse_train[:,0]
        rmse_test = posrmse_test[:,0]



        print w_pos, ' w_pos'
        print posfx_test, ' fx test'
        #print posrmse_test, ' posrmse test'
        #print pos_tau, ' pos_tau'

        print fx_train, '  fx_train'



        print accept_ratio




        print 'sucessfully sampled'

        burnin = 0.1 * num_samples  # use post burn in samples


        fx_mu = fx_test.mean(axis=0)
        fx_high = np.percentile(fx_test, 95, axis=0)
        fx_low = np.percentile(fx_test, 5, axis=0)

        fx_mu_tr = posfx_train[:,0,:].mean(axis=0)
        fx_high_tr = np.percentile(posfx_train[:,0,:], 95, axis=0)
        fx_low_tr = np.percentile(posfx_train[:,0,:], 5, axis=0)

        rmse_tr = scipy.mean(rmse_train[int(burnin):])
        rmsetr_std = np.std(rmse_train[int(burnin):])
        rmse_tes = scipy.mean(rmse_test[int(burnin):])
        rmsetest_std = np.std(rmse_test[int(burnin):])

        print rmse_tr, rmsetr_std, rmse_tes, rmsetest_std

        np.savetxt('mcmcresults/results.txt', (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')

        #ytestdata = testdata[:, input]
        #ytraindata = traindata[:, input]

        plt.plot(x_test, y_test, label='actual')
        plt.plot(x_test, fx_mu, label='pred. (mean)')
        plt.plot(x_test, fx_low, label='pred.(5th percen.)')
        plt.plot(x_test, fx_high, label='pred.(95th percen.)')
        plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Plot of Test Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/mcmcrestest.png')
        plt.savefig('mcmcresults/mcmcrestest.svg', format='svg', dpi=600)
        plt.clf()
        # -----------------------------------------
        plt.plot(x_train, y_train, label='actual')
        plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
        plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
        plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
        plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Plot of Train Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/mcmcrestrain.png')
        plt.savefig('mcmcresults/mcmcrestrain.svg', format='svg', dpi=600)

        plt.clf()

        plt.hist(rmse_train, bins=np.linspace(0, 0.1, num=20))  # plt.hist passes it's arguments to np.histogram
        plt.title("RMSE train")
        plt.savefig('mcmcresults/rmsetrain.png')

        plt.clf()

        plt.hist(rmse_train, bins=np.linspace(0, 0.1, num=20))  # plt.hist passes it's arguments to np.histogram
        plt.title("RMSE train")
        plt.savefig('mcmcresults/rmsetest.png')
        plt.clf()


if __name__ == "__main__": main()
