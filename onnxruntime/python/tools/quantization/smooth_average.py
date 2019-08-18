# Computing a smooth average (weighted average of data split by change points)

import random
import numpy as np

def compute_CUSUM(input_list):
    '''
    Calculates cumulative sums given an input list
        parameter input_list: 1D array of floats
        return: 1D array of floats
    '''
    avg = np.average(input_list)
    cusum_list = [0]
    for i in range(len(input_list)):
        cusum_list.append(cusum_list[i] + input_list[i] - avg)
    return cusum_list

def bootstrap(n, units_list):
    '''
    Performs bootstraps on an input list
        parameter n: int, number of bootstraps
        parameter units_list: 1D array of floats
        return: 2D array containing CUSUMs of n shuffled versions of units_list
    '''
    bootstraps, shuffledCUSUMS = [], []
    for i in range(n):
        bootstraps.append(random.sample(units_list, len(units_list)))
    for shuffle in bootstraps:
        shuffle_CUSUM = compute_CUSUM(shuffle)
        shuffledCUSUMS.append(shuffle_CUSUM)
    return shuffledCUSUMS

def confidence_level(n, units_list):
    '''
    Calculates confidence of a change point occurring
        parameter n: int, number of bootstraps
        parameter units_list: 1D array of floats
        return: double, confidence level
    '''
    units_list_CUSUM = compute_CUSUM(units_list)
    units_list_diff = max(units_list_CUSUM) - min(units_list_CUSUM) # S0diff
    diffs, bootstraps = [], []
    x = 0 # number of bootstraps for which S0diff < Sdiff
    for i in range(n):
        bootstraps.append(random.sample(units_list, len(units_list)))
    for shuffle in bootstraps:
        shuffle_CUSUM = compute_CUSUM(shuffle)
        diffs.append(max(shuffle_CUSUM) - min(shuffle_CUSUM))
    for i in diffs:
        if i < units_list_diff:
            x += 1
    return x/n

# MSE (mean square error) formula
def get_x1_bar(input_list, m):
    '''
    Intermediate step in MSE calculation
        parameter input_list: 1D array of floats
        parameter m: int
        return: average of input_list summed up to m elements
    '''
    summation = 0;
    for i in range(1, m+1):
        summation += input_list[i-1]
    xbar1 = summation/m
    return xbar1

def get_x2_bar(input_list, m):
    summation = 0
    length = len(input_list)
    for i in range(m+1, length+1):
        summation += input_list[i-1]
    xbar2 = summation/(length - m)
    return xbar2

def getMSE1(input_list, m):
    '''
    Intermediate step in MSE calculation
        parameter input_list: 1D array of floats
        parameter m: int
        return: sum fromm 1 to m of list elements' deviation from average, i.e. MSE
    '''
    summation = 0
    for i in range(1, m+1):
        summation += (input_list[i-1] - get_x1_bar(input_list, m))**2
    return summation

def getMSE2(input_list, m):
    summation = 0;
    length = len(input_list)
    for i in range(m+1, length+1):
        summation += (input_list[i-1] - get_x2_bar(input_list, m))**2
    return summation

def getMSE(input_list, m):
    '''
    Calculates MSE
        parameter input_list: 1D array of floats
        parameter m: int
        return: final MSE value associated with each m
    '''
    summation = getMSE1(input_list, m) + getMSE2(input_list, m)
    return summation

def transform(input_list):
    '''
    Calculates the element position in input_list that minimizes MSE as the
    best estimator of the last point before a change
        parameter input_list: 1D array of floats
        return: int, index of input_list corresponding to minimum MSE
    '''
    MSEList = []
    for m in range(1, len(input_list)+1):
        MSEList.append(getMSE(input_list, m))
    minMSEIndex = MSEList.index(min(MSEList))
    return minMSEIndex

def smooth_average(input_list, bootstraps=1000, confidence_threshold=0.9):
    '''
    Calculates weighted (smooth) average of input_list, where weights are
    assigned by volume contained between discovered change points
        parameter input_list: 1D array of floats
        parameter bootstraps: number of random re-orderings of data
        parameter confidence_threshold: threshold determining whether change is significant
        return: smooth average
    '''
    input_list_copy = input_list
    input_list_length = len(input_list_copy)
    change_points = []
    def change_point_analysis(input_list, input_list_copy):
        conf = confidence_level(bootstraps, input_list)
        if len(input_list) <= 1 or conf < confidence_threshold:
            return
        else:
            index = transform(input_list)
            change_points.append(input_list_copy.index(input_list[index-1]) + 1) # positions in input_list_copy
            bottom_list = input_list[:index]
            top_list = input_list[index:]
            if len(bottom_list) <= 1 or len(top_list) <= 1:
                return
            change_point_analysis(bottom_list, input_list_copy)
            change_point_analysis(top_list, input_list_copy)
    change_point_analysis(input_list, input_list_copy)
    markings = sorted(set(change_points)) # last points before change
    markings.insert(0, 0)
    markings.append(input_list_length)
    smooth_average = 0
    for i in range(1, len(markings)):
        smooth_average += ((markings[i] - markings[i-1])/input_list_length)*np.average(input_list_copy[markings[i-1]:markings[i]])
    return smooth_average
