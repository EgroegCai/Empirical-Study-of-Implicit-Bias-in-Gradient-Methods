import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import Classifier
from trainer import Trainer


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

if __name__ == '__main__':
    epochs = 4000
    batch_size = 4
    use_cuda = False#torch.cuda.is_available()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    img_size = (32,32,3)

    classifier = Classifier(img_size)
    print(classifier)




    # get some random training images
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(),lr=0.001,momentum=0.9)


    # Train model
    f = open('../data/DNN/result.txt','w')
    tempf = open('../data/DNN/classification.txt','w')

    trainer = Trainer(classifier,optimizer,criterion,use_cuda=use_cuda)
    print(trainer)
    for epoch in range(epochs):
        print("epoch = %d" % epoch)
        train_loss = trainer.train_epoch(train_loader)#,save_training_gif=True)
        # L2_norm
        norm = list(classifier.parameters())[-2].norm(2)        
        print('Norm of last hidden layer: %f' % norm)
        # training loss        
        print('Training Loss: %f' % train_loss)
        # training error
        train_class_correct = 0.0
        train_class_total = 0.0
        tempf.write("epoch %f\n" %(epoch))
        tempf.write("training predictions\n")
        tempf.flush()
        for data in train_loader:
            images, labels = data
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = classifier(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            tempf.write(predicted.numpy().__repr__())     
            tempf.write("\n")  
            tempf.flush() 
            train_class_correct += (predicted == labels).sum().item()
            train_class_total += batch_size
        train_err = 1.0 - train_class_correct / train_class_total
        print('Training error: %f %%' % (100*train_err))

        tempf.write("testing predictions\n")
        tempf.flush()
        # test loss
        test_loss = 0.0
        for _,data in enumerate(test_loader,0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.data[0]
        test_loss = test_loss / len(test_loader)
        print("Test_loss: %f" % test_loss)
        # validation accuracy
        test_class_correct = 0.0
        test_class_total = 0.0
        for data in test_loader:
            images, labels = data
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()            
            outputs = classifier(Variable(images))

            _, predicted = torch.max(outputs.data, 1)
            tempf.write(predicted.numpy().__repr__()) 
            tempf.write("\n")         
            tempf.flush()      
            test_class_correct += (predicted == labels).sum().item()
            test_class_total += batch_size
        test_err = 1.0 - test_class_correct / test_class_total
        print('Test error: %f %%' % (100*test_err))
        f.write('%f,%f,%f,%f,%f\n' % (norm, train_loss, train_err, test_loss, test_err))
        f.flush()
        
    print('Finished Training')    
    f.close()
    tempf.close()
    checkpoint = [50,100,200,400,2000,4000]

    detailer = iter(test_loader)
    images, labels = detailer.next()

    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # outputs = classifier(Variable(images))
    print('DONE')