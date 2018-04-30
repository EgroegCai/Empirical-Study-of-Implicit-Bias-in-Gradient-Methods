import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Trainer():
    def __init__(self, classifier, optimizer, loss, print_every=50, use_cuda=False):
        self.classifier = classifier
        self.opt = optimizer
        self.loss = loss
        self.use_cuda = use_cuda
        self.print_every = print_every

        if self.use_cuda:
            self.classifier.cuda()

    def train(self,train_loader,epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i,data in enumerate(train_loader,0):
                inputs,labels = data

                inputs, labels = Variable(inputs), Variable(labels)

                self.opt.zero_grad()
                outputs = self.classifier(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.opt.step()

                running_loss += loss.data[0]
                if i % 2000 == 1999:
                    print('[%d, %5d] Train loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')