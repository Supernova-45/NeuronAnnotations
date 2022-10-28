from napari.layers import Points
import napari
import skimage.data
import skimage.filters
from napari.types import PointsData
import pandas as pd
import numpy as np

from magicgui import magicgui

from enum import Enum
from pathlib import Path

# modify Points class


class FixedPoints(Points):

    def _move(self):
        """Points are not allowed to move."""
        pass

def napari_csv_to_array(filename):
    """
    Input: Napari CSV file and resolution; Output: NumPy array
    """
    df = pd.read_csv(filename, usecols=['axis-0', 'axis-1', 'axis-2', 'label'])
    
    return df.to_numpy()

def sub_arrays(filename):
    points = napari_csv_to_array(filename)
    subArr = [[] for x in range(15)]
    for point in points:
        subArr[int(point[3])].append([point[0], point[1], point[2]])

    return np.array(subArr, dtype="object")

def neurons_and_not(filename):
    points = napari_csv_to_array(filename)
    subArr = [[],[]]
    for point in points:
        index = 1 if point[3] >= 1 else 0
        subArr[index].append([point[0], point[1], point[2]])
        
    return np.array(subArr, dtype="object")    

def labeled_viewer(csvName, imageName):
    labeledPoints = neurons_and_not(csvName)

    colors = ['white','red']
    
    with napari.gui_qt():
        image_path = imageName

        viewer = napari.Viewer()
        viewer.open(image_path, scale=None)

        for i in range(2):
            viewer.add_points(labeledPoints[i], ndim= 3, size=2,
                              face_color=colors[i],edge_width = 0)
            
def colored_proximity_viewer(csvName,imageName):
    labeledPoints = sub_arrays(csvName)

    colors = ['white',
              'red',
              'pink'
              'bop orange',
              'bop purple',
              'cyan',
              'blue',
              'gray',
              'green',
              'yellow',
              'magenta',
              'gist_earth',
              'hsv',
              'inferno',
              'magma',
              'plasma',
              'turbo',
              'twilight',
              'viridis',
              ]
    
    with napari.gui_qt():
        image_path = imageName

        viewer = napari.Viewer()
        viewer.open(image_path, scale=None)

        for i in range(12):
            viewer.add_points(labeledPoints[i], ndim= 3, size=2,
                              face_color=colors[i],edge_width = 0)
            
def main():
    # colored_proximity_viewer("data/local_max_pairings_p1_s1.csv","/Users/alexandrakim/Desktop/BUGS2022/2P/Left_Forebrain_STD_1907029_9dpf_bigvsmall_52z_2P.tif")
    labeled_viewer("data/local_max_pairings_p1_s1.csv","/Users/alexandrakim/Desktop/BUGS2022/2P/Left_Forebrain_STD_1907029_9dpf_bigvsmall_52z_2P.tif")
    


if __name__ == "__main__":
    main()
