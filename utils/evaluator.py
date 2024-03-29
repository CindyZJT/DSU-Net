import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.ndimage import _ni_support
from scipy.spatial.distance import cdist
from skimage import measure
from medpy.metric.binary import hd95


class Evaluator_dice(object):
    def __init__(self):
        self.diceAll = 0.
        self.num = 0
        self.dice_channels = [0]*2
        self.num_channels = [0]*2
        self.current_dice = 0

    def add_volume(self, pred, target, channel=None):
        assert(pred.max() <= 1 and target.max() <= 1, 'Input should be bool type')
        intersection = (pred * target).sum()
        current_dice = (2. * intersection) / (pred.sum() + target.sum()) if (pred.sum() + target.sum()) != 0 else 1
        pred_pixel = np.array(np.where(pred > 0))
        target_pixel = np.array(np.where(target > 0))
        if channel is not None:
            self.dice_channels[channel] += current_dice
            self.num_channels[channel] += 1
        self.diceAll += current_dice
        self.current_dice = current_dice
        self.num += 1
        return pred_pixel.shape[1], target_pixel.shape[1]

    def get_eval(self, channel=None):
        assert(self.num != 0)
        if channel is None:
            # return self.diceAll/self.num
            return self.current_dice
        else:
            return self.dice_channels[channel]/self.num_channels[channel]


class Evaluator_hd95(object):
    def __init__(self):
        self.hd95All = 0
        self.num = 0
        self.hd95_channels = [0] * 2
        self.num_channels = [0] * 2

    def add_volume(self, result, reference, channel = None, voxelspacing=None, connectivity=1):
        """
        95th percentile of the Hausdorff Distance.
        Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
        images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
        commonly used in Biomedical Segmentation challenges.
        Parameters
        ----------
        result : array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
        reference : array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
        voxelspacing : float or sequence of floats, optional
            The voxelspacing in a distance unit i.e. spacing of elements
            along each dimension. If a sequence, must be of length equal to
            the input rank; if a single number, this is used for all axes. If
            not specified, a grid spacing of unity is implied.
        connectivity : int
            The neighbourhood/connectivity considered when determining the surface
            of the binary objects. This value is passed to
            `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
            Note that the connectivity influences the result in the case of the Hausdorff distance.
        Returns
        -------
        hd : float
            The symmetric Hausdorff Distance between the object(s) in ```result``` and the
            object(s) in ```reference```. The distance unit is the same as for the spacing of
            elements along each dimension, which is usually given in mm.
        See also
        --------
        :func:`hd`
        Notes
        -----
        This is a real metric. The binary images can therefore be supplied in any order.
        """
        hd1 = self.surface_distances(result, reference, voxelspacing, connectivity)
        hd2 = self.surface_distances(reference, result, voxelspacing, connectivity)
        hd95 = np.percentile(np.hstack((hd1, hd2)), 95)

        pred_pixel = np.array(np.where(result > 0))
        target_pixel = np.array(np.where(reference > 0))

        if channel is not None:
            self.hd95_channels[channel] += hd95
            self.num_channels[channel] += 1

        self.hd95All += hd95
        self.num += 1

        return pred_pixel.shape[1], target_pixel.shape[1]


    def surface_distances(self, result, reference, voxelspacing=None, connectivity=1):
        """
        The distances between the surface voxel of binary objects in result and their
        nearest partner surface voxel of a binary object in reference.
        """
        result = np.atleast_1d(result.astype(np.bool))
        reference = np.atleast_1d(reference.astype(np.bool))
        a = reference.max()
        if voxelspacing is not None:
            voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
            voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
            if not voxelspacing.flags.contiguous:
                voxelspacing = voxelspacing.copy()

        # binary structure
        footprint = generate_binary_structure(result.ndim, connectivity)

        # test for emptiness
        if 0 == np.count_nonzero(result):
            raise RuntimeError('The first supplied array does not contain any binary object.')
        if 0 == np.count_nonzero(reference):
            raise RuntimeError('The second supplied array does not contain any binary object.')

            # extract only 1-pixel border line of objects
        result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
        reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

        # compute average surface distance
        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
        sds = dt[result_border]
        return sds

    def get_eval(self, channel=None):
        assert (self.num != 0)
        if channel is None:
            return self.hd95All/self.num
        else:
            return self.hd95_channels[channel]/self.num_channels[channel]

class Evaluator_hd95V2(object):
    def __init__(self):
        self.hd95All = 0
        self.num = 0
        self.hd95_channels = [0] * 2
        self.num_channels = [0] * 2

    def add_volume(self, pred, target, channel):

        res_hd95 = hd95(pred, target)

        pred_pixel = np.array(np.where(pred > 0))
        target_pixel = np.array(np.where(target > 0))
        if channel is not None:
            self.hd95_channels[channel] += res_hd95
            self.num_channels[channel] += 1

        self.hd95All += res_hd95
        self.num += 1

        return pred_pixel.shape[1], target_pixel.shape[1]


    def get_eval(self, channel=None):
        assert (self.num != 0)
        if channel is None:
            return self.hd95All/self.num
        else:
            return self.hd95_channels[channel]/self.num_channels[channel]