# from image_manipulation import imageManipulation, arrayManipulation
import napari
from cropper import Cropper, Slices

import tifffile as tif
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def napari_csv_to_array(filename):
    """
    Input: Napari CSV file and resolution; Output: NumPy array
    """
    df = pd.read_csv(filename, usecols=['axis-0', 'axis-1', 'axis-2'])
    df = df.astype(int) # losing info??

    return df.to_numpy()

def removeNotCropped(croppedArr, statusArr):
    count = 0
    while count < len(statusArr):
        if statusArr[count][0] == False:
            croppedArr = np.delete(croppedArr,count,axis=0)
            statusArr = np.delete(statusArr,count,axis=0)
            count -= 1
            
        count += 1
        
    return croppedArr
        
def main():
    
    im = tif.imread("/Users/alexandrakim/Desktop/BUGS2022/2P/Left_Forebrain_STD_1907029_9dpf_bigvsmall_52z_2P.tif")
    notNeurons = napari_csv_to_array("data/2P_lindsey_suhan_not_neurons.csv")
    neurons = napari_csv_to_array("data/2P_lindsey_suhan_neurons.csv")
    
    slice = Slices(np.array([1,9,9]))
    notNeuronArr, notNeuronStatus = slice.crop(notNeurons,im,True)
    notNeuronArr = removeNotCropped(notNeuronArr, notNeuronStatus)
    notNeuronLabels = np.zeros((len(notNeuronArr),1),dtype='int')

    neuronArr, neuronStatus = slice.crop(neurons,im,True)
    neuronArr = removeNotCropped(neuronArr, neuronStatus)
    neuronLabels = np.ones((len(neuronArr),1),dtype='int')

    print(neuronArr.shape)
    print(notNeuronArr.shape)

    data = np.concatenate((neuronArr,notNeuronArr),axis=0)
    labels = np.concatenate((neuronLabels,notNeuronLabels),axis=0)

    #shuffle
    shuffled_indices = np.random.permutation(len(data))
    data = data[shuffled_indices]
    labels = labels[shuffled_indices]
    
    
    np.savez('data/2P_lindsey_suhan_data.npz', data = data, labels = labels)
    '''
    fig, ax = plt.subplots(4, 4)

    for row in range(4):
        for col in range(4):
            plt.subplot(4, 4, row*4+col+1)
            ax[row][col].set_axis_off()
            plt.imshow(neuronArr[np.random.randint(0, 900)],interpolation='nearest')
    '''

    plt.show()
    # tif.imwrite("data/example_slice.tif",notNeuronArr[3], photometric='minisblack')

if __name__ == "__main__":
    main()
