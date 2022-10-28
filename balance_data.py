"""
Balances labeled data containing neurons and not neurons by undersampling
"""

import napari
import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

def binary_csv_to_data(filename):
    """
    Returns ZYX points from a Numpy array and their labels (0 = not neuron, 1 = neuron)
    """
    
    dfZYX = pd.read_csv(filename, usecols=['axis-0', 'axis-1', 'axis-2'])
    dfLabel = pd.read_csv(filename, usecols=['label'])
    
    npLabel = dfLabel.to_numpy()
    for index, label in enumerate(npLabel):
        npLabel[index] = 1 if label >= 1 else 0
        
    return dfZYX.to_numpy(), npLabel

def colorful_csv_to_data(filename):
    """
    Returns ZYX points from a Numpy array and their labels 
    (0 = not neuron, 1 = neuron found on first pass, 2 = neuron found on second pass, 
    3 = neuron found on 3rd+ pass)
    """
    
    dfZYX = pd.read_csv(filename, usecols=['axis-0', 'axis-1', 'axis-2'])
    dfLabel = pd.read_csv(filename, usecols=['label'])
    
    npLabel = dfLabel.to_numpy()
    for index, label in enumerate(npLabel):
        if label == 0:
            npLabel[index] = 0
        elif label == 1:
            npLabel[index] = 1
        elif label == 2:
            npLabel[index] = 2
        else:
            npLabel[index] = 3
        
    return dfZYX.to_numpy(), npLabel

def napari_csv_to_array(filename):
    """
    Input: Napari CSV file and resolution; Output: NumPy array
    """
    df = pd.read_csv(filename, usecols=['axis-0', 'axis-1', 'axis-2'])
    return df.to_numpy()

def bar_plot(labelArr, quantityArr, time):
    """
    Bar graph
    labelArr: Neurons with 0, 1, 2, 3+, etc.
    quantityArr: (int) Number of neurons with corresponding label
    Time: before or after undersampling
    """
    
    plt.bar(labelArr, quantityArr, color ='blue', width = 0.5)
    plt.ylabel("Size of dataset")
    plt.title(f"Local max data {time} undersampling")
    
    plt.show()
    
def labeled_viewer(arr, imageName):
    """
    Displays points colored white = not neuron, red = label 1, blue = label 2, teal = label 3+
    arr: 3D array [[][][][]] where each [] = label with shape (n,3)
    containing ZYX points 
    imageName: (String) filename of image to display
    """
    
    colors = ['white','red','blue','pink']
    
    with napari.gui_qt():
        image_path = imageName

        viewer = napari.Viewer()
        viewer.open(image_path, scale=None)

        for i in range(4):
            viewer.add_points(arr[i], ndim= 3, size=2,
                              face_color=colors[i],edge_width = 0)

def labeled_arr(X,y):
    """
    Separates labeled points into a format which can be fed to labeled_viewer() 
    X, y = outputs of RandomUnderSampler.fit_resample()
    X: array of XYZ points
    y: 1D array with shape (1,n)
    """
    
    subArr = [[],[],[],[]]
    for index, value in enumerate(y):
        subArr[int(value) if value <= 2 else 3].append(X[index])
        
    return np.array(subArr, dtype="object")    


def main():
    # neurons = napari_csv_to_array("data/2P_neurons.csv")
    # notNeurons = napari_csv_to_array("data/2P_not_neurons.csv")
     
    X, y = colorful_csv_to_data("data/2P_pairings_p1_s1.csv")
    quantity = [np.count_nonzero(y==0), np.count_nonzero(y==1), 
                np.count_nonzero(y==2), np.count_nonzero(y >= 3)]
    undersample = RandomUnderSampler(sampling_strategy={0:(quantity[1]+50), 1:quantity[1]})
    Xresample, yresample = undersample.fit_resample(X, y)
    
    undersampledArr = labeled_arr(Xresample, yresample)
    labeled_viewer(undersampledArr,"/Users/alexandrakim/Desktop/BUGS2022/2P/Left_Forebrain_STD_1907029_9dpf_bigvsmall_52z_2P.tif")
    '''
    X, y = colorful_csv_to_data("data/local_max_pairings_p1_s1.csv")
    
    """
    # bar chart before undersampling
    label = ["not neurons", "1", "2", "3+"]
    quantity = [np.count_nonzero(y==0), np.count_nonzero(y==1), 
                np.count_nonzero(y==2), np.count_nonzero(y >= 3)]
    bar_plot(label,quantity,"before")
    """
    quantity = [np.count_nonzero(y==0), np.count_nonzero(y==1), 
                np.count_nonzero(y==2), np.count_nonzero(y >= 3)]
    undersample = RandomUnderSampler(sampling_strategy={0:(quantity[1]+50), 1:quantity[1]})
    Xresample, yresample = undersample.fit_resample(X, y)
    
    #"""
    # bar chart after undersampling
    label = ["not neurons", "1", "2", "3+"]
    quantity = [np.count_nonzero(yresample==0), np.count_nonzero(yresample==1), 
                np.count_nonzero(yresample==2), np.count_nonzero(yresample >= 3)]
    bar_plot(label,quantity,"after")
    # """
    # view with napari
    undersampledArr = labeled_arr(Xresample, yresample)
    labeled_viewer(undersampledArr,"/Users/alexandrakim/Desktop/BUGS2022/2P/Left_Forebrain_STD_1907029_9dpf_bigvsmall_52z_2P.tif")
    '''
    
if __name__ == "__main__":
    main()