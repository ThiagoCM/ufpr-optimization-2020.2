from scipy import optimize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from random import uniform
import evaluation_functions as ef


def rosenbrock(x, y):
    return (x-1)**2 + 100*(y-x**2)**2;

def beale(x, y):
    return (x*y + 1.5 - x)**2 + (x * (y**2) + 2.5 - x)**2 + (x * (y**3) + 2.625 - x)

def booth(x, y):
    return (x + 2 * y - 7)**2 + (2*x  + y - 5)**2

def matyas(x, y):
    return 0.26*(x**2 + y**2) - 0.48*(x*y)


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
    answerList = []
    answerValuesList = []
    for i in range(30):
        print('\nInteration:',i)
        x0 = [uniform(minValue, maxValue), uniform(minValue, maxValue)]
        print('Initial Point:', x0)
        answer = optimize.minimize(optimize.rosen, x0, method='BFGS', jac = optimize.rosen_der,
                                   options={'disp':True, 'return_all':True})
        print('Rosenbrock Minimized Value:',answer.x)
        answerList.append(answer)
        answerValuesList.append(answer.x)
    return answerList, answerValuesList

def bfgs_rosenbrock30d(minValue, maxValue):
    answerList = []
    answerValuesList = []
    for i in range(30):
        print('\nInteration:',i)
        x0 = []
        for j in range(30):
            x0.append(uniform(minValue, maxValue))
        print('Initial Point:', x0)
        answer = optimize.minimize(optimize.rosen, x0, method='BFGS', jac = optimize.rosen_der,
                                   options={'disp':True, 'return_all':True})
        print('Rosenbrock Minimized Value:',answer.x)
        answerList.append(answer)
        answerValuesList.append(answer.x)
    return answerList, answerValuesList

#plot_rosenbrock()
result2dList, result2dValueList = bfgs_rosenbrock2d(-1,1)
minimum2dInterationResult = ef.minimum_iterations(result2dList)
minimum2dInterationConvergeResult = ef.minimum_iterations_converge(result2dList)
ef.evaluate_method(result2dValueList, 'Rosenbrock 2D')

result30dList, result30dValueList = bfgs_rosenbrock30d(-1, 1)
minimum30dInterationResult = ef.minimum_iterations(result30dList)
minimum30dInterationConvergeResult = ef.minimum_iterations_converge(result30dList)
ef.evaluate_method(result30dValueList , 'Rosenbrock 30D')