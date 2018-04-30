from torch.autograd import Variable


class Trainer():
    def __init__(self, classifier, optimizer, loss_fn, print_every=2000, use_cuda=False):
        self.classifier = classifier
        self.opt = optimizer
        self.loss_fn = loss_fn
        self.use_cuda = use_cuda
        self.print_every = print_every

        if self.use_cuda:
            self.classifier.cuda()

    def train_epoch(self,train_loader):
        running_loss = 0.0
        total_loss = 0.0
        count = 0
        for i,data in enumerate(train_loader,0):
            count += 1
            inputs,labels = data

            inputs, labels = Variable(inputs), Variable(labels)
            if self.use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            self.opt.zero_grad()
            outputs = self.classifier(inputs)
            if self.use_cuda:
                outputs = outputs.cuda()
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.opt.step()
            total_loss += loss.data[0]

            running_loss += loss.data[0]
            if i % self.print_every == self.print_every-1:
                print('[%5d] Train loss: %.3f' %
                    (i + 1, running_loss / 2000))
                running_loss = 0.0          
        return total_loss / count