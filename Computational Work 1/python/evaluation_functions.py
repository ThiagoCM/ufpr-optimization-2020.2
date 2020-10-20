# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 23:35:11 2020

@author: 624590
"""

import numpy as np


def minimum_iterations(resultList):
    minimumFunctionValue = 1000000.000
    minimumIteration = 1000000
    for result in resultList:
        if result.fun < minimumFunctionValue and result.nit < minimumIteration:
            minimumFunctionValue = result.fun
            minimumIteration = result.nit
            finalResult = result
    return finalResult

def minimum_iterations_converge(resultList):
    minimumFunctionValue = 1000000.000
    minimumIterationConverge = 1000000
    for result in resultList:
        if result.fun < minimumFunctionValue and result.nfev < minimumIterationConverge:
            minimumFunctionValue = result.fun
            minimumIterationConverge = result.nfev
            finalResult = result
    return finalResult

def evaluate_method(answerList, methodName):
    print('\nEvaluation of:', methodName)
    print('\nMinimum:', np.min(answerList, axis=0))
    print('\nMaximum:', np.max(answerList, axis=0))
    print('\nMean:', np.mean(answerList, axis=0))
    print('\nMedian:', np.median(answerList, axis=0))