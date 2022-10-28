"""
Adopted from Anna Nadtochiy: https://github.com/LemonJust/sycatch/blob/main/sycatch/cropper.py
"""
import numpy as np


class Cropper:
    """
    Takes care of cropping 3D volumes or 2D slices from an image, going through the center.
    """

    def __init__(self, shape):
        """
        shape : [z, y, x] size of the slice, in pixels
        centroids : [z, y, x] center of the slice, in pixels
        img: 3D numpy array for an image
        """
        # shape as a 3d and a 2d array (one dimension is 1)
        assert np.sum(shape == 1) <= 1, "Shape must be 3D or 2D : no dimensions of size 1," \
                                        " or exactly one such dimension"
        assert np.sum(shape == 0) < 1, "Shape must be 3D or 2D : no 0 dimensions"

        self.shape_3d = np.array(shape)

    def crop(self, centroids, img, return_status=False):
        """
        img: 3D array , image from which to crop
        centroids : N x 3 array
        Returns
        imgs: array N_dim1_dim2 with image slices through the centroids or all zeros if crop was not successful
        crop_status: weather or not the centroid was cropped
        """
        n_centroids = centroids.shape[0]
        imgs = np.zeros((n_centroids, self.shape_3d[0], self.shape_3d[1], self.shape_3d[2]))
        crop_status = np.ones((n_centroids, 1)).astype('bool')

        for i, centroid in enumerate(centroids):
            start = centroid - self.shape_3d // 2
            end = centroid + self.shape_3d // 2 + 1
            # check that crop is fully inside the image:
            if np.any(start < 0) or np.any(end > img.shape):
                crop_status[i] = False
            else:
                imgs[i, :] = img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        if return_status:
            # return images with dimension of size 1 (if 2d case) removed and a list of cropped and not
            return np.squeeze(imgs), crop_status
        else:
            return np.squeeze(imgs)

    def get_croppable(self, centroids, img_shape, as_idx=False):
        """
        Returns a boolean list, with false if the centroid is too close to the boarder of the image
        and can't be cropped.
        as_idx : if True will return the indexes of cropable centroids , if False will return a boolean array
        """

        start = centroids - self.shape_3d // 2
        end = centroids + self.shape_3d // 2 + 1
        # check that volumes are fully inside the image:
        cropable = ~np.logical_or(np.any(start < 0, axis=1),
                                  np.any(end > img_shape, axis=1))
        if as_idx:
            return np.where(cropable)[0]
        else:
            return cropable


class Slices(Cropper):
    """
    Takes care of 2D slices from an image, going through the center.
    """

    def __init__(self, shape):
        """
        shape : [z, y, x] size of the slice, in pixels
        centroids : [z, y, x] center of the slice, in pixels
        img: 3D numpy array for an image
        """
        # TODO : make it possible to initialise with a 2d shape + orientation

        super().__init__(shape)
        self.shape_2d = self.get_shape_2d()
        self.orientation = self.get_orientation()

    def get_shape_2d(self):
        """
        Returns 2d shape by dropping the 0 dimension.
        """
        if np.sum(self.shape_3d == 1) == 1:
            return self.shape_3d[self.shape_3d != 1]
        else:
            return None

    def get_orientation(self):
        """
        Names the orientation based on shape.
        """
        # figure out slice orientation by looking at  what dimension is missing
        orient_list = ['yx', 'zx', 'zy']
        is_0 = np.where(self.shape_3d == 1)[0][0]
        return orient_list[is_0]

    def flip(self, dim):
        """
        flips slices for data augmentation.
        dim: dimension to flip , 0 or 1
        """
        pass

    def rotate_90(self):
        """
        Rotates slices by 90 deg for data augmentation. Only for 'xy' slices
        """
        pass


class Volumes(Cropper):
    """
    Takes care of 3D blocks from an image, going through the center.
    """

    def __init__(self, shape):
        """
        shape : [z, y, x] size of the volume, in pixels
        centroids : [z, y, x] centers of the volumes, in pixels
        img: 3D numpy array for an image
        """
        super().__init__(shape)
        self.orientation = '3d'

    def flip(self, dim):
        """
        flips slices for data augmentation.
        dim: dimension to flip , 0, 1 or 2
        """
        pass

    def rotate_90(self):
        """
        Rotates volumes by 90 deg around z axis for data augmentation ( rotation in XY plane)
        """
        pass