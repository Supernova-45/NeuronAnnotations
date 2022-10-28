'''
Program that compares segmenters' data annotations from a Napari/Fiji CSV file.
'''

from token import SLASH
from turtle import shape
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from matplotlib_venn import venn3, venn2

def napari_to_array(filename, xRes, yRes, zRes):
    """
    Input: Napari CSV file and resolution; Output: NumPy array
    """
    df = pd.read_csv(filename,usecols= ['axis-0','axis-1','axis-2'])
    df.columns = ['Z','Y','X']
    df = df[['X','Y','Z']]
    df['Z'] *= zRes # is this a universal measurement?
    df['X'] *= xRes
    df['Y'] *= yRes
    return df.to_numpy()

def fiji_to_array(filename, xRes, yRes, zRes):
    """
    Input: Fiji CSV file and resolution in pixels per micrometer; Output: NumPy array
    """
    df = pd.read_csv(filename,usecols= ['X','Y','Slice'])
    df['Slice'] -= 1 # fiji slices start at 1; napari starts at 0
    df['Slice'] *= 4.8 # slices to um
    # doesn't fiji default to um already?
    df['X'] /= 1 
    df['Y'] /= 1 
    return df.to_numpy()

def to_labeled_csv(arr,filename):
    df = pd.DataFrame(arr)
    df = df.reset_index()
    df.columns = ['index','axis-2','axis-1','axis-0','label']
    df['axis-0'] = np.round_(df['axis-0'] / 4.8, decimals = 1)
    df = df[['index','axis-0','axis-1','axis-2','label']]

    df.to_csv(filename, sep=',',index=None)
    
def to_napari_csv(arr, filename):
    """
    Turns 2D XYZ array of points into Napari CSV format
    Input: NumPy array, String (name of output file)
    Output: CSV file called "filename"
    """
    df = pd.DataFrame(arr)
    df = df.reset_index()
    df.columns = ['index','axis-2','axis-1','axis-0']
    df['axis-0'] = np.round_(df['axis-0'] / 4.8, decimals = 1)
    df = df[['index','axis-0','axis-1','axis-2']]

    df.to_csv(filename, sep=',',index=None)

def plot(arr):
    """
    Plots a 3D array of data points in Matplotlib
    Inputs: A 2D array; each entry is [arr, plotColor, labelName] 
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for info in arr:
        ax.scatter3D(info[0][:,0],info[0][:,1],info[0][:,2],
                    s=20,color=info[1],label=info[2])

    fig.canvas.set_window_title('Ok3')
    ax.set_title("Zebrafish brain neurons", fontweight = 'bold')
    ax.set_xlabel('X-axis', fontweight = 'bold')
    ax.set_ylabel('Y-axis', fontweight = 'bold')
    ax.set_zlabel('Z-axis (slice)', fontweight = 'bold')

    plt.legend(loc="upper right")
    plt.show()

def show_venn3(tuple,label1,label2,label3,color1,color2,color3,alpha):
    """
    Input: tuple (a,b,ab,c,ac,bc,abc); three labels, colors = strings
    Output: Weighted venn diagram with three circles
    """
    venn3(subsets = tuple, set_labels = (label1,label2,label3), 
        set_colors = (color1,color2,color3), alpha = alpha)
    plt.show()
    
def show_venn2(tuple, label1, label2, color1, color2, alpha):
    """
    Input: tuple (a,b,ab); two labels and colors = strings
    Output: Weighted venn diagram with two circles
    """
    venn2(subsets = tuple, set_labels = (label1,label2), 
        set_colors = (color1,color2), alpha = alpha)
    plt.show()  

def nearest_pairs(v1, v2, radius):
    """
    Adopted from synspy: https://github.com/informatics-isi-edu/synspy.git
    Find nearest k-dimensional point pairs between v1 and v2 and return via output arrays.

       Inputs:
         v1: array with first pointcloud with shape (n, k)
         kdt1: must be cKDTree(v1) for correct function
         v2: array with second pointcloud with shape (m, k)
         radius: maximum euclidean distance between points in a pair

       Use greedy algorithm to assign nearest neighbors without
       duplication of any point in more than one pair.

       Outputs:
         out1: for each point in kdt1, gives index of paired point from v2 or -1
         iv1_for_v2: out2: for each point in v2, gives index of paired point from v1 or -1

    """
    depth = min(max(v1.shape[0], v2.shape[0]), 100)

    out1 = np.full((v1.shape[0], 1), -1)
    out2 = np.full((v2.shape[0], 1), -1)

    kdt1 = cKDTree(v1)
    dx, pairs = kdt1.query(v2, depth, distance_upper_bound=radius)
    for d in range(depth):
        for idx2 in np.argsort(dx[:, d]):
            if dx[idx2, d] < radius:
                if out2[idx2] == -1 and out1[pairs[idx2, d]] == -1:
                    out2[idx2] = pairs[idx2, d]
                    out1[pairs[idx2, d]] = idx2
    iv1_for_v2 = out2
    return out1, iv1_for_v2

def overlap_size(arr): 
    """
    Finds the number of non -1 points from a 1D array
    Input: 1D array; Output: int
    """
    return len(arr) - np.count_nonzero(arr == -1)

def percent_matched(arr1, arr2, radius):
    """
    Computes the percent of points clicked by 2 segmenters and the percent of points which only one segmenter clicked
    Input: two 3D NumPy arrays; Output: Float between 0 and 100
    """
    closestOne, closestTwo = nearest_pairs(arr1, arr2, radius)
    matched = (overlap_size(closestOne) / 
                (len(arr1) + len(arr2) - overlap_size(closestOne)))
    mismatched1 = ((len(arr1) - overlap_size(closestOne)) / 
                (len(arr1) + len(arr2) - overlap_size(closestOne)))
    mismatched2 = ((len(arr2) - overlap_size(closestOne)) / 
                (len(arr1) + len(arr2) - overlap_size(closestOne)))
    return round(matched*100,4), round(mismatched1*100,4), round(mismatched2*100,4)

def two_segs(arr1, arr2, radius, nameOne, nameTwo):
    """
    Returns information about the points clicked by two segmenters
    Input: Numpy arrays from two segmenters, tolerance threshold, name of two segmenters (String)
    Output: String
    """
    matched12, mismatched12, mismatched21 = percent_matched(arr1, arr2, radius)
    msg = f"Seg {nameOne} and seg {nameTwo} overlapped by {matched12} percent, with {mismatched12} percent mismatched by seg {nameOne} and {mismatched21} percent mismatched by seg {nameTwo}.\n"
    return msg

def three_segs(arr1, arr2, arr3, radius):
    """
    Returns information about combinations of three segmenters
    Input: Numpy arrays from two segmenters, tolerance threshold, name of two segmenters (String)
    Output: String
    """
    msg = two_segs(arr1, arr2, radius, "1", "2") + two_segs(arr1, arr3, radius, "1", "3") + two_segs(arr2, arr3, radius, "2", "3")
    return msg

def union(arr1, arr2, radius):
    """
    Returns an array of unique points (the union) from 2 arrays
    Input: 2 XYZ arrays and a threshold; Output: XYZ array
    """
    union = arr1.copy()
    x, y = nearest_pairs(arr1, arr2, radius)
    for index in range(len(y)):
        if (y[index] == -1):
            union = np.append(union, [arr2[index]], axis=0)
            
    return union

def venn_two_sizes(arr1, arr2, radius):
    """
    Calculates various unions of two sets
    Input: Two XYZ NumPy arrays and a tolerance threshold 
    Output: Tuple of (a,b,ab)
    """
    x, y = nearest_pairs(arr1, arr2, radius)
    ab = overlap_size(x)
    a = 0
    for index in x:
        if index == -1:
            a += 1
            
    b = 0
    for index in y:
        if index == -1:
            b += 1

    return (a, b, ab)

def venn_three_sizes(arr1, arr2, arr3, radius):
    """
    Calculates various unions of three sets
    Input: Three XYZ NumPy arrays and a tolerance threshold 
    Output: Tuple of (a,b,ab,c,ac,bc,abc)
    """
    x, y = nearest_pairs(arr1, arr2, radius)
    ab = overlap_size(x)
    overlap = np.empty((ab,3))
    count = 0
    for index in x:
        if index != -1:
            overlap[count] = arr2[index]
            count += 1
    x, y = nearest_pairs(overlap, arr3, radius)
    abc = overlap_size(x)

    x, y = nearest_pairs(arr1, arr3, radius)
    ac = overlap_size(x)
    x, y = nearest_pairs(arr2, arr3, radius)
    bc = overlap_size(x)

    # By principle of inclusion-exclusion
    a = len(arr1)-ab-ac+abc
    b = len(arr2)-ab-bc+abc
    c = len(arr3)-ac-bc+abc

    return (a,b,ab-abc,c,ac-abc,bc-abc,abc)

def many_segs(*args):
    pass

def label_local_max(localMax, annotators, radius, filename):
    """
    Returns a csv with 0 for Not a neuron and 1 for Yes a neuron
    Input: XYZ array of local maxes and annotator points, tolerance threshold (double), filename (string)
    """
    a, b = nearest_pairs(localMax, annotators, radius)
    neuronLabel = np.full((localMax.shape[0],1), 0) # 1D array containing 1s and 0s
    
    for index in range(len(a)):
        if (a[index] > -1):
            neuronLabel[index] = 1
    
    df = pd.DataFrame(localMax)
    df = df.reset_index()
    df.columns = ['index','X','Y','Z']
    df['Z'] = np.round_(df['Z'] / 4.8, decimals = 1)
    df['label'] = neuronLabel
    
    df.to_csv(filename, sep=',',index=None)

def plot_prominence(annotators):
    
    for s in range(0,6):
        prominence = []
        truePos = []
        falsePos = []
        falseNeg = []
        
        for i in range(1,11):
        
            localMax = fiji_to_array("data/local_max/local_max_2P_prominence_"+str(i)+
                                     "_sigma_"+str(s)+".csv", 1, 1, 4.8)
            a,b,ab = venn_two_sizes(annotators,localMax,4.5)
            prominence.append(i)
            truePos.append(ab)
            falsePos.append(b)
            falseNeg.append(a)
            
        plt.subplot(2,3,s+1)
        plt.xlabel("Sigma " + str(round(s*0.4,1)))
        ax = plt.gca()
        ax.set_ylim([0, 2300])
        plt.plot(prominence, truePos, '-o',label='True pos')
        plt.plot(prominence, falsePos, '-o',label='False pos')
        plt.plot(prominence, falseNeg, '-o',label='False neg')
        plt.legend()
        
    # plt.title("Local Max vs. Annotators' Points")
    
    plt.show()
    
def test_comparison():
    '''
    Testing the resolution aspect of nearest_pairs().
    '''
    arrOne = napari_to_array("data/test_pairing_depth_1.csv",1,1,1)
    arrTwo = napari_to_array("data/test_pairing_depth_2.csv",1,1,1)
    
    print("Resolution of [1,1,1]:")
    print("Array 1: \n" + str(arrOne))
    print("Array 2: \n" + str(arrTwo))
    print("Nearest pairs within radius of 5:")
    print(nearest_pairs(arrOne, arrTwo,5) + "\n")
    
    arrOne = napari_to_array("data/test_pairing_depth_1.csv",1.17,1.17,4.8)
    arrTwo = napari_to_array("data/test_pairing_depth_2.csv",1.17,1.17,4.8)
    
    print("Resolution of [1.17,1.17,4.8]:")
    print("Array 1: \n" + str(arrOne))
    print("Array 2: \n" + str(arrTwo))
    print("Nearest pairs within radius of 5:")
    print(nearest_pairs(arrOne, arrTwo,5) + "\n")

def all_pairing(segs, maxes, radius):
    """
    Iteratively finds all the pairs within a certain radius and labels them 
    Input: segmenter array, max array, radius
    Output: max array with each element labeled 1-n
    """
    
    indices = np.arange(0,len(maxes))
    unpairedMaxi = maxes.copy()
    zeroCol = np.zeros((len(maxes),1))
    labeledMaxi = np.append(maxes,zeroCol,axis=1)
    done = False
    removed = 0

    while not done:
        v1, v2 = nearest_pairs(segs,unpairedMaxi,radius)
        if overlap_size(v2) != 0:
            removed += 1
            i = 0
            while i < len(v2):
                if v2[i] != -1:
                    labeledMaxi[indices[i]][3] = removed
                    unpairedMaxi = np.delete(unpairedMaxi,i,axis=0)
                    indices = np.delete(indices,i)
                    v2 = np.delete(v2,i,axis=0)
                    i -= 1
                
                i += 1
        else:
            done = True
                                
    return labeledMaxi
 
def split_by_label(arr):
    labeled = [[] for i in range(10)]
    
    for i in range(len(arr)):
        labeled[int(arr[i][3])].append(arr[i].tolist())
        
    return labeled
        
    
def main():
    seg1 = napari_to_array("data/seg1_points.csv",1,1,4.8)
    seg2 = napari_to_array("data/seg2_points.csv",1,1,4.8)
    
    # label_local_max(seg1,seg2, 4.5, 'test')
    # print(two_segs(seg1, seg2, 2, "1", "2"))

    # suhan = fiji_to_array("data/suhan_7_9_2022.csv",1)
    # lindsey = np.concatenate((napari_to_array("data/lindsey_sn_7_9_2022.csv",1.7), napari_to_array("data/lindsey_mn_7_9_2022.csv",1.7)), axis=0)
    # alex = napari_to_array("data/alex_7_9_2022.csv", 1.7)

    # one = napari_to_array("/Users/alexandrakim/Desktop/BUGS2022/napari_2P_point.csv", 1,1,4.8)
    # two = fiji_to_array("/Users/alexandrakim/Desktop/BUGS2022/fiji_2P_point.csv", 1,1,4.8)
    
    # show_venn2(venn_two_sizes(seg1,seg2,4),'one','two','blue','cyan',0.5)
    
    suhan = fiji_to_array("data/suhan_2P_7_19_2022.csv",1,1,4.8)
    lindsey = np.concatenate((napari_to_array("data/lindsey_2P_mn_7_19_22.csv",1,1,4.8), napari_to_array("data/lindsey_2P_sn_7_19_22.csv",1,1,4.8)), axis=0)
    alex = napari_to_array("data/alex_2P_7_19_22.csv", 1,1,4.8)
    
    ls = union(lindsey,suhan, 5)
    # sla = union(sl, alex, 4.5) # all 2P neurons annotated
    
    # to_napari_csv(sla, "data/all_annotators_2P.csv")

    localMax = fiji_to_array("data/local_max/local_max_2P_prominence_1_sigma_1.csv", 1, 1, 4.8)

    paired = all_pairing(ls, localMax, 5)
    #plot([[paired, 'red', 'maxes'],[sla, 'green', 'neurons']])

    
    to_labeled_csv(paired,"data/lindsey_suhan_pairings_p1_s1.csv")

    # print(split_by_label(paired))
    # plot_prominence(sla)
    
    # plot([[suhan,'red','Annotators'],[paired,'blue','Local max']])

    # label_local_max(paired,suhan, 4.5, 'data/local_max/local_max_labeled.csv')
    # print(two_segs(sla, localMax, 4.5, 'Annotators', 'Local max'))

    # show_venn2(venn_two_sizes(sla,localMax,4.5),'Annotators','Local max','blue','cyan',0.6)

    # plot([[suhan, 'red', 'Suhan'],[lindsey, 'green', 'Lindsey'],[alex, 'blue', 'Alex']])

    # to_napari_csv((np.concatenate((suhan,lindsey, alex), axis=0)), "data/all_annotators_2P")
    
    # show_venn3((venn_three_sizes(alex, lindsey, suhan, 6.5)), 'Alex','Lindsey','Suhan', 'purple','blue','cyan',0.5)

    # print(three_segs(alex, lindsey, suhan, 4))


    # fiji = fiji_to_array("/Users/alexandrakim/Desktop/BUGS2022/fiji_two_points.csv", 1)
    # napari = napari_to_array("/Users/alexandrakim/Desktop/BUGS2022/napari_two_points.csv", 1.17)

    # to_napari_csv(fiji, "data/output_points.csv")
    # plot([[fiji, 'red', 'fiji'],[napari, 'green','napari']])

if __name__ == "__main__":
    main()