import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from numpy import genfromtxt

axis_font = {'fontname':'Arial', 'size':'12'}


data = genfromtxt('../data/DNN/result.txt', delimiter=',')


def plot_data(data):
    L2_norm = data[:,0]
    Train_loss = data[:,1]
    Train_err = data[:,2]
    Test_loss = data[:,3]
    Test_err = data[:,4]
    Epochs = range(len(data[:,]))
    
    print('Plotting L2norm...')
    plt.figure(1)
    plt.semilogx(Epochs,L2_norm)
    plt.title('L2 Norm of the Last Weight Layer')
    plt.xlabel('Epochs',**axis_font)
    plt.ylabel('L2 Norm',**axis_font)
    plt.savefig('../data/DNN/L2norm.png')

    print('Plotting Traing and Testing loss...')
    plt.figure(2)
    plt.loglog(Epochs,Train_loss,label='Training_loss')
    plt.loglog(Epochs,Test_loss,label='Testing_loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epochs',**axis_font)
    plt.ylabel('Loss',**axis_font)
    plt.legend()
    plt.savefig('../data/DNN/Loss.png')

    print('Plotting Traing and Testing error...')
    plt.figure(3)
    plt.semilogx(Epochs,Train_err,label='Training_error')
    plt.semilogx(Epochs,Test_err,label='Testing_error')
    plt.title('Training and Testing Error')
    plt.xlabel('Epochs',**axis_font)
    plt.ylabel('Error/%',**axis_font)
    plt.legend()
    plt.savefig('../data/DNN/Error.png')
    print('Done.')

if __name__ == '__main__':
    plot_data(data)
