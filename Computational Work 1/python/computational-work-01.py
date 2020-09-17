from scipy import optimize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from random import uniform
def rosenbrock(x, y):
    return (x-1)**2 + 100*(y-x**2)**2;

def plot_rosenbrock():
    step = .15
    X = np.arange(-2, 2, step)
    Y = np.arange(-1, 3, step)
    X, Y = np.meshgrid(X, Y)

    Z = rosenbrock(X, Y)
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca(projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cm.gist_heat_r, linewidth=0)
    ax.set_zlim(0, 2000)
    fig.colorbar(surface, shrink=0.5, aspect=10)
    plt.show()

def bfgs_rosenbrock2d(minValue, maxValue):
    answerListValues = []
    for i in range(30):
        print('\nInteration:',i)
        x0 = [uniform(minValue, maxValue), uniform(minValue, maxValue)]
        print('Initial Point:', x0)
        answer = optimize.minimize(optimize.rosen, x0, method='BFGS', jac = optimize.rosen_der,
                                   options={'disp':True, 'return_all':True})
        print('Rosenbrock Minimized Value:',answer.x)
        answerListValues.append(answer.x)
    return answerListValues

def bfgs_rosenbrock30d(minValue, maxValue):
    answerListValues = []
    for i in range(30):
        print('\nInteration:',i)
        x0 = []
        for j in range(30):
            x0.append(uniform(minValue, maxValue))
        print('Initial Point:', x0)
        answer = optimize.minimize(optimize.rosen, x0, method='BFGS', jac = optimize.rosen_der,
                                   options={'disp':True, 'return_all':True})
        print('Rosenbrock Minimized Value:',answer.x)
        answerListValues.append(answer.x)
    return answerListValues


def evaluate_method(answerList, methodName):
    print('\nEvaluation of:', methodName)
    print('\nMinimum:', np.min(answerList, axis=0))
    print('\nMaximum:', np.max(answerList, axis=0))
    print('\nMean:', np.mean(answerList, axis=0))
    print('\nMedian:', np.median(answerList, axis=0))

#plot_rosenbrock()
answersValues = bfgs_rosenbrock2d(-1,1)
evaluate_method(answersValues, 'Rosenbrock 2D')

answersValues = bfgs_rosenbrock30d(-1, 1)
evaluate_method(answersValues, 'Rosenbrock 30D')