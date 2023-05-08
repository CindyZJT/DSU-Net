
import pydensecrf.densecrf as dcrf
import torch
import os

from pydensecrf.utils import create_pairwise_gaussian
# from pydensecrf.utils import unary_from_softmax


import numpy as np


def dence_crf(prediction, image):

    prediction = prediction.numpy()
    shape = image.shape

    unary = unary_from_softmax(prediction)

    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(np.prod(shape), 7)

    d.setUnaryEnergy(unary)

    feats = create_pairwise_gaussian(sdims=(1, 1, 1), shape=shape)

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    #
    # feats = create_pairwise_bilateral(sdims=(50, 50, 50), schan=(20, 20, 20),
    #                                   img=image, chdim=3)

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)

    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1], image.shape[2]))
    return torch.from_numpy(res)


def unary_from_softmax(sm, scale=None, clip=1e-5):
    """Converts softmax class-probabilities to unary potentials (NLL per node).

    Parameters
    ----------
    sm: numpy.array
        Output of a softmax where the first dimension is the classes,
        all others will be flattend. This means `sm.shape[0] == n_classes`.
    scale: float
        The certainty of the softmax output (default is None).
        If not None, the softmax outputs are scaled to range from uniform
        probability for 0 outputs to `scale` probability for 1 outputs.
    clip: float
        Minimum value to which probability should be clipped.
        This is because the unary is the negative log of the probability, and
        log(0) = inf, so we need to clip 0 probabilities to a positive value.
    """
    num_cls = sm.shape[0]
    if scale is not None:
        assert 0 < scale <= 1, "`scale` needs to be in (0,1]"
        uniform = np.ones(sm.shape) / num_cls
        sm = scale * sm + (1 - scale) * uniform
    if clip is not None:
        sm = np.clip(sm, clip, 1.0)
    return -np.log(sm).reshape([num_cls, -1]).astype(np.float32)



def create_pairwise_bilateral(sdims, schan, img, chdim=-1):
    """
    Util function that create pairwise bilateral potentials. This works for
    all image dimensions. For the 2D case does the same as
    `DenseCRF2D.addPairwiseBilateral`.

    Parameters
    ----------
    sdims: list or tuple
        The scaling factors per dimension. This is referred to `sxy` in
        `DenseCRF2D.addPairwiseBilateral`.
    schan: list or tuple
        The scaling factors per channel in the image. This is referred to
        `srgb` in `DenseCRF2D.addPairwiseBilateral`.
    img: numpy.array
        The input image.
    chdim: int, optional
        This specifies where the channel dimension is in the image. For
        example `chdim=2` for a RGB image of size (240, 300, 3). If the
        image has no channel dimension (e.g. it has only one channel) use
        `chdim=-1`.

    """
    # Put channel dim in right position
    if chdim == -1:
        # We don't have a channel, add a new axis
        im_feat = img[np.newaxis].astype(np.float32)
    else:
        # Put the channel dim as axis 0, all others stay relatively the same
        im_feat = np.rollaxis(img, chdim).astype(np.float32)

    # scale image features per channel
    # Allow for a single number in `schan` to broadcast across all channels:
    if isinstance(schan, Number):
        im_feat /= schan
    else:
        for i, s in enumerate(schan):
            im_feat[i] /= s

    # create a mesh
    cord_range = [range(s) for s in im_feat.shape[1:]]
    mesh = np.array(np.meshgrid(*cord_range, indexing='ij'), dtype=np.float32)

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        mesh[i] /= s

    feats = np.concatenate([mesh, im_feat])
    return feats.reshape([feats.shape[0], -1])


