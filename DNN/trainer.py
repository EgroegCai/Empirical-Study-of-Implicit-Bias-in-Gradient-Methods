import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Trainer():
    def __init__(self, classifier, optimizer, print_every=50, use_cuda=False):
        self.NN = classifier
        self.N_opt = optimizer
        self.use_cuda = use_cuda
        self.print_every = print_every

        if self.use_cuda:
            self.NN.cuda()

    def train(self,data,epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i,data in enumerate(trainloader,0):
                inputs,labels = data

                inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')