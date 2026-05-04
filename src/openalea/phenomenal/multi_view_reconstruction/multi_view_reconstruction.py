# -*- python -*-
#
#       Copyright INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
# ==============================================================================

import collections
import cv2
import scipy.spatial
import numpy

from ..object import VoxelGrid, ImageView

# ==============================================================================
# Class

Voxels = collections.namedtuple("Voxels", ["position", "size"])

# ==============================================================================
# deprecated numpy-python slow integral (kept as explicit 'doc' of what integral image is for us)


def get_integrale_image(img):
    a = numpy.zeros_like(img, dtype=int)
    a[img > 0] = 1
    for y, x in numpy.ndindex(a.shape):
        if x - 1 >= 0:
            a[y, x] += a[y, x - 1]
        if y - 1 >= 0:
            a[y, x] += a[y - 1, x]
        if x - 1 >= 0 and y - 1 >= 0:
            a[y, x] -= a[y - 1, x - 1]
    return a


def integral_image(input_array):
    binary = (input_array > 0).astype(numpy.uint8)
    integral = cv2.integral(binary, sdepth=cv2.CV_32S)
    return integral[1:, 1:]


def get_voxels_corners(voxels_position, voxels_size):
    """According to the voxels position and their size, return a numpy array
    containing for each input voxels the position of the 8 corners.

    Parameters
    ----------
    voxels_position : numpy.ndarray
        Center position of the voxels

    voxels_size : float
        Diameter size of the voxels

    Returns
    -------
    a : numpy.array
    """

    r = voxels_size / 2.0

    x_minus = voxels_position[:, 0] - r
    x_plus = voxels_position[:, 0] + r
    y_minus = voxels_position[:, 1] - r
    y_plus = voxels_position[:, 1] + r
    z_minus = voxels_position[:, 2] - r
    z_plus = voxels_position[:, 2] + r

    a1 = numpy.column_stack((x_minus, y_minus, z_minus))
    a2 = numpy.column_stack((x_plus, y_minus, z_minus))
    a3 = numpy.column_stack((x_minus, y_plus, z_minus))
    a4 = numpy.column_stack((x_minus, y_minus, z_plus))
    a5 = numpy.column_stack((x_plus, y_plus, z_minus))
    a6 = numpy.column_stack((x_plus, y_minus, z_plus))
    a7 = numpy.column_stack((x_minus, y_plus, z_plus))
    a8 = numpy.column_stack((x_plus, y_plus, z_plus))

    a = numpy.concatenate((a1, a2, a3, a4, a5, a6, a7, a8), axis=1)
    a = numpy.reshape(a, (a.shape[0] * 8, 3))

    return a


def get_bounding_box_voxel_projected(voxels_position, voxels_size, projection):
    """Compute the bounding box value according the radius, angle and
    calibration parameters of point_3d projection

    Parameters
    ----------
    voxels_position : numpy.ndarray
        Center position of voxel

    voxels_size : float
        Size of side geometry of voxel

    projection : function ((x, y, z)) -> (x, y)
        Function of projection who take 1 argument (tuple of position (x, y, z))
        and return this position 2D (x, y)

    Returns
    -------
    bbox : numpy.ndarray
        [[x_min, x_max, y_min, y_max], ...]
        Containing min and max value of point_3d projection in x and y axes.
    """

    voxels_corners = get_voxels_corners(voxels_position, voxels_size)

    pt = projection(voxels_corners)

    pt = numpy.reshape(pt, (pt.shape[0] // 8, 8, 2))

    bbox = numpy.column_stack((pt.min(axis=1), pt.max(axis=1)))

    return bbox


# ==============================================================================


def split_voxels_in_eight(voxels):
    """Split each voxel in 8 en return the numpy.array position

          _ _ _ _ _ _ _ _ _                              _ _ _ _ _ _ _ _ _
        /                  /|                          /        /         /|
       /                  / |                         /--------/---------/ |
      /_ _ _ _ _ _ _ _ _ /  |                        /_ _ _ _ / _ _ _ _ /| |
     |                  |   |                       |        |         | |/|
     |                  |   |       =======>        |        |         | / |
     |                  |   |                       |_ _ _ _ | _ _ _ _ |/| |
     |                  |  /                        |        |         | |/
     |                  | /                         |        |         | /
     | _ _ _ _ _ _ _ _ _|/                          |_ _ _ _ | _ _ _ _ |/



    Parameters
    ----------
    voxels : Voxels
        Where position is a numpy.array([[x, y, z], ...] containing the center
        position of each voxel and size the diameter size value (float) of each
        voxel.

    Returns
    -------
    voxels : Voxels
    """
    if len(voxels.position) == 0:
        return Voxels(voxels.position, voxels.size / 2.0)

    r = voxels.size / 4.0

    x_minus = voxels.position[:, 0] - r
    x_plus = voxels.position[:, 0] + r
    y_minus = voxels.position[:, 1] - r
    y_plus = voxels.position[:, 1] + r
    z_minus = voxels.position[:, 2] - r
    z_plus = voxels.position[:, 2] + r

    a1 = numpy.column_stack((x_minus, y_minus, z_minus))
    a2 = numpy.column_stack((x_plus, y_minus, z_minus))
    a3 = numpy.column_stack((x_minus, y_plus, z_minus))
    a4 = numpy.column_stack((x_minus, y_minus, z_plus))
    a5 = numpy.column_stack((x_plus, y_plus, z_minus))
    a6 = numpy.column_stack((x_plus, y_minus, z_plus))
    a7 = numpy.column_stack((x_minus, y_plus, z_plus))
    a8 = numpy.column_stack((x_plus, y_plus, z_plus))

    return Voxels(
        numpy.concatenate((a1, a2, a3, a4, a5, a6, a7, a8), axis=0),
        voxels.size / 2.0
    )


# ==============================================================================


def voxels_is_visible_in_image(
    voxels_position, voxels_size, image, projection, inclusive, image_int=None
):
    """
    Return a numpy array containing True if the voxel are
        projected is photo-consistent on image else False

    **Algorithm**

    1. Project each voxel center position on the image, if the position
    projected (x, y) is positive on image return True

    |

    2. If the bounding box of the voxel projected have positive value on
    the image the voxel are True

    |

    3. Check if one pixel containing in the bounding box projected on image
       have positive value, if yes return True else return False

    Parameters
    ----------
    voxels_position : numpy.array([[x, y, z], ...]
        Center position of the voxels

    voxels_size : float
        diameter size of the voxels

    image: numpy.array
        Binary image where the voxels are projected.

    projection : function (numpy.array([[x, y, z], ...]) -> numpy.array([[x, y], ...])
        Function of projection who take 1 argument (numpy.array([[x, y, z],
        ...] of voxels positions) and return the projected 2D position
        numpy.array([[x, y], ...])

    inclusive: Describe if the voxels projection are out of the image,
    they are considered like still visible

    image_int: Integral image of the binary image (optimization)


    Returns
    -------
    out : numpy.array([True, False, ...])
        Numpy array containing True if the voxel are
        projected is photo-consistent on image else False
    """

    height, length = image.shape
    ori_result = numpy.zeros(
        (
            len(
                voxels_position,
            )
        ),
        dtype=int,
    )

    r = projection(voxels_position)

    cond = (r[:, 0] >= 0) & (r[:, 1] >= 0) & (r[:, 0] < length) & (r[:, 1] < height)

    rr = r[cond].astype(int)

    (ori_result[cond])[image[rr[:, 1], rr[:, 0]] > 0] = 1
    not_cond = numpy.logical_not(ori_result > 0)

    result = ori_result[not_cond]
    voxels_position = voxels_position[not_cond]

    # ==========================================================================

    min_xy_max_xy = get_bounding_box_voxel_projected(
        voxels_position, voxels_size, projection
    )

    vv = (
        (min_xy_max_xy[:, 2] < 0)
        | (min_xy_max_xy[:, 0] >= length)
        | (min_xy_max_xy[:, 3] < 0)
        | (min_xy_max_xy[:, 1] >= height)
    )

    not_vv = numpy.logical_not(vv)
    result[vv] = 1 if inclusive else 0

    min_xy_max_xy = min_xy_max_xy[not_vv]
    bb = result[not_vv]

    min_xy_max_xy = numpy.floor(min_xy_max_xy).astype(int)

    # X limit
    (min_xy_max_xy[:, 0])[min_xy_max_xy[:, 0] >= length] = length - 1
    (min_xy_max_xy[:, 2])[min_xy_max_xy[:, 2] >= length] = length - 1
    # Y limit
    (min_xy_max_xy[:, 1])[min_xy_max_xy[:, 1] >= height] = height - 1
    (min_xy_max_xy[:, 3])[min_xy_max_xy[:, 3] >= height] = height - 1
    # Under zero limit
    min_xy_max_xy[:, 0:2] -= 1  # For integral image optimization
    min_xy_max_xy[min_xy_max_xy < 0] = 0

    # ==========================================================================

    for i, (x_min, y_min, x_max, y_max) in enumerate(min_xy_max_xy):
        if (
            image_int[y_max, x_max]
            + image_int[y_min, x_min]
            - image_int[y_min, x_max]
            - image_int[y_max, x_min]
        ) > 0:
            bb[i] = 1

    # for i, (x_min, y_min, x_max, y_max) in enumerate(min_xy_max_xy):
    #     if numpy.count_nonzero(image[y_min:y_max + 1, x_min:x_max + 1]) > 0:
    #         bb[i] = 1

    result[not_vv] = bb
    ori_result[not_cond] = result

    return ori_result


# ==============================================================================
def reconstruction_grid(center=(0.0, 0.0, 0.0), grid_size=4096, voxel_size=512):
    """
    Setup a reconstruction grid
    Args:
        center: coordinates of the center of the grid
        grid_size: outer edge length of the grid
        voxel_size: edge length of voxels composing the grid

    Returns:
        a Voxels (positions, size) named tuple
    """
    voxels_position = numpy.array([center])
    voxels = Voxels(voxels_position, grid_size)
    while voxels.size != voxel_size:
        voxels = split_voxels_in_eight(voxels)
    return voxels


def check_each(image_views, check=True):
    """Return a check dict for all keys in image_views according to check

    Args:
        image_views : {name: ImageView, ...}
         Dict of phenomenal.object.ImageView objects gathering image, projection, where image is a binary image
         (numpy.ndarray) and projection a function projecting (x, y, z) ->  (u, v) coordinate on image
        check: bool | str | [str,...]
         If True or False, the value associated to each key present in image_views. If a list of name,
         the names to be set to True
    """
    if isinstance(check, bool):
        return {k: check for k in image_views}

    check = [check] if isinstance(check, str) else check

    return {
        name: any(name.startswith(cam) for cam in check)
        for name in image_views
    }


def filter_voxels(voxels, image_views, error_tolerance=0, clear_outside=True):
    """Filter voxels, keeping the one photo_consistent with all minus error_tolerance images in image_views"""

    clear_view = check_each(image_views, clear_outside)
    photo_consistent_score = numpy.zeros((len(voxels.position),), dtype=int)
    for i, (k, image_view) in enumerate(image_views.items()):
        if image_view.integral is None:
            image_view.integral = integral_image(image_view.image)
        inclusive = not clear_view[k]
        photo_consistent_score += voxels_is_visible_in_image(
            voxels.position,
            voxels.size,
            image_view.image,
            image_view.projection,
            inclusive,
            image_view.integral
        )
        cond = photo_consistent_score >= i + 1 - error_tolerance
        voxels = Voxels(voxels.position[cond], voxels.size)
        photo_consistent_score = photo_consistent_score[cond]

    return voxels


def tolerant_reconstruction(image_views, voxels_size=4, error_tolerance=0, clear_outside=None, start=None):

    if start is None:
        voxels = reconstruction_grid()
    elif isinstance(start, VoxelGrid):
        voxels = Voxels(start.voxels_position, start.voxels_size)
    else:
        voxels = start

    while voxels.size > voxels_size:
        if len(voxels.position) == 0:
            break
        voxels = split_voxels_in_eight(voxels)
        voxels = filter_voxels(voxels, image_views, error_tolerance=error_tolerance, clear_outside=clear_outside)

    return voxels


def multi_tolerant_reconstruction(image_views, voxels_size=4, max_tolerance=0, clear_outside=None, start=None):
    voxels = tolerant_reconstruction(image_views, voxels_size=voxels_size, error_tolerance=0, clear_outside=clear_outside,
                                     start=start)
    not_reconstructed = {}
    for k, iv in image_views.items():
        projected = project_voxel_centers_on_image(
            voxels.position,
            voxels.size,
            iv.image.shape,
            iv.projection,
        )
        view = ImageView(numpy.logical_not(projected)*255,
                            iv.projection)
        not_reconstructed[k] = view

    for i in range(max_tolerance):
        new_voxels = reconstruction_grid()
        while new_voxels.size > voxels_size:
            if len(new_voxels.position) == 0:
                break
            new_voxels = split_voxels_in_eight(new_voxels)
            new_voxels = filter_voxels(new_voxels, image_views, error_tolerance=i + 1, clear_outside=clear_outside)
            new_voxels = filter_voxels(new_voxels, not_reconstructed, error_tolerance=0, clear_outside=clear_outside)

        if len(new_voxels.position) == 0:
            break

        for k, iv in image_views.items():
            projected = project_voxel_centers_on_image(
                new_voxels.position,
                new_voxels.size,
                iv.image.shape,
                iv.projection,
            )
            not_reconstructed[k].image *= numpy.logical_not(projected)

        voxels = Voxels(numpy.concatenate((voxels.position, new_voxels.position), axis=0), voxels.size)

    return voxels


def reconstruction_3d(
    image_views,
    voxels_size=4,
    error_tolerance=0,
    start=None,
    clear_outside=True
):
    """
    Construct a list of voxel represented object with positive value on binary
    image in images of images_projections.

    Parameters
    ----------

    image_views : {name: ImageView, ...}
        Dict of phenomenal.object.ImageView objects gathering image, projection, where image is a binary image
        (numpy.ndarray) and projection a function projecting (x, y, z) ->  (u, v) coordinate on image

    voxels_size : float, optional
        Edge length of reconstructed voxels

    error_tolerance : int, optional
        Control the degree of consistency of the reconstructed object with its projected views
        If not provided the reconstruction is fully consistent with all views (error_tolerance=0)
        If positive, the number of inconsistent views tolerated per voxel
        If negative, a composite 3d reconstruction is computed, iteratively aggregating reconstructions of different
        tolerance, from zero to abs(error_tolerance), discarding at each step voxels projecting on projections of
        already reconstructed

    start : VoxelGrid | [Voxel,..], optional
        A voxel grid to be used as starting point.
        If None (default), a grid returned by defaults of openalea.phenomenal.multiview_reconstruction.reconstruction_grid

    clear_outside: bool | str | [str,...], optional
        Should voxels projected outside image_views be kept ? True (default) or False set a unique rule for all views.
        if a list of name is provided, only image_views whose key starts with names are used to clear voxels


    Returns
    -------
    out : VoxelGrid
    """

    if len(image_views) == 0:
        raise ValueError("Images views is empty")

    if error_tolerance >= 0:
        voxels = tolerant_reconstruction(image_views, voxels_size=voxels_size, error_tolerance=error_tolerance,
                                           start=start, clear_outside=clear_outside)
    else:
        voxels = multi_tolerant_reconstruction(image_views, voxels_size=voxels_size, max_tolerance=abs(error_tolerance),
                                           start=start, clear_outside=clear_outside)

    return VoxelGrid(voxels.position, voxels.size)

# ==============================================================================


def project_voxel_centers_on_image(
    voxels_position, voxels_size, shape_image, projection, value=255, dtype=numpy.uint8
):
    """
    Create a image with same shape that shape_image and project each voxel on
    image and write positive value (255) on it.

    Parameters
    ----------
    voxels_position : numpy.ndarray
        Voxels center position [[x, y, z], ...]
    voxels_size : float
        Diameter size of the voxels
    shape_image: 2-tuple
        Size height and width of the image target projected
    projection : function ((x, y, z)) -> (x, y)
        Function of projection who take 1 argument (tuple of position (x, y, z))
         and return this position 2D (x, y)
    value : int
        value between 0 and 255 of positive pixel. By default, 255.
    dtype : type
        numpy type of the returned image. By default, numpy.uint8.

    Returns
    -------
    out : numpy.ndarray
        Binary image
    """
    height, length = shape_image
    img = numpy.zeros((height, length), dtype=dtype)

    min_xy_max_xy = get_bounding_box_voxel_projected(
        voxels_position, voxels_size, projection
    )

    vv = (
        (min_xy_max_xy[:, 2] < 0)
        | (min_xy_max_xy[:, 0] >= length)
        | (min_xy_max_xy[:, 3] < 0)
        | (min_xy_max_xy[:, 1] >= height)
    )

    not_vv = numpy.logical_not(vv)
    min_xy_max_xy = min_xy_max_xy[not_vv]

    min_xy_max_xy = numpy.floor(min_xy_max_xy)
    min_xy_max_xy[min_xy_max_xy < 0] = 0
    (min_xy_max_xy[:, 0])[min_xy_max_xy[:, 0] >= length] = length - 1
    (min_xy_max_xy[:, 1])[min_xy_max_xy[:, 1] >= height] = height - 1
    (min_xy_max_xy[:, 2])[min_xy_max_xy[:, 2] >= length] = length - 1
    (min_xy_max_xy[:, 3])[min_xy_max_xy[:, 3] >= height] = height - 1
    min_xy_max_xy = min_xy_max_xy.astype(int)

    for x_min, y_min, x_max, y_max in min_xy_max_xy:
        img[y_min : y_max + 1, x_min : x_max + 1] = value

    return img


def project_voxels_position_on_image(
    voxels_position, voxels_size, shape_image, projection
):
    """
    Create an image with same shape that shape_image and project each voxel on
    image and write positive value (255) on it.

    Parameters
    ----------
    voxels_position : [(x, y, z)]
        cList (collections.deque) of center position of voxel

    voxels_size : float
        Size of side geometry of voxel

    shape_image: Tuple
        size height and length of the image target projected

    projection : function ((x, y, z)) -> (x, y)
        Function of projection who take 1 argument (tuple of position (x, y, z))
         and return this position 2D (x, y)

    Returns
    -------
    out : numpy.ndarray
        Binary image
    """

    voxels_position = numpy.array(voxels_position)
    height, length = shape_image
    img = numpy.zeros((height, length), dtype=numpy.uint8)

    voxels_corners = get_voxels_corners(voxels_position, voxels_size)
    pt = projection(voxels_corners)
    pt = numpy.reshape(pt, (pt.shape[0] // 8, 8, 2))

    pt[pt < 0] = 0
    (pt[:, :, 0])[pt[:, :, 0] >= length] = length - 1
    (pt[:, :, 1])[pt[:, :, 1] >= height] = height - 1
    pt = numpy.floor(pt).astype(int)
    for points in pt:
        hull = scipy.spatial.ConvexHull(points)
        cv2.fillConvexPoly(img, points[hull.vertices], 255)

    return img


# ==============================================================================


def image_error(img_ref, img_src, precision=2):
    """
    Return false position and true negative result from the comparison on
    two binaries images
    """

    img_ref = img_ref.astype(numpy.int32)
    nb_ref = max(numpy.count_nonzero(img_ref), 1)
    nb_ref2 = max(numpy.count_nonzero(img_ref == 0), 1)
    img_src = img_src.astype(numpy.int32)
    img = numpy.subtract(img_ref, img_src)
    true_negative = numpy.bitwise_and(img_ref == 0, img_src == 0)
    true_negative = round(
        numpy.count_nonzero(true_negative) * 100.0 / nb_ref2, precision
    )
    # true_negative = numpy.bitwise_and(img_ref == 0, img_src == 0)

    false_positive = round(
        numpy.count_nonzero(img[img < 0]) * 100.0 / nb_ref2, precision
    )
    false_negative = round(
        numpy.count_nonzero(img[img > 0]) * 100.0 / nb_ref, precision
    )
    print(true_negative, false_positive, false_negative)
    return false_positive, false_negative


def reconstruction_error(voxels_grid, image_views):
    """
    Compute the reconstruction error (false positive and true negative) of
    the 3d reconstruction from the image view.

    Parameters
    ----------
    voxels_grid: VoxelGrid
        The voxel grid

    image_views : numpy.ndarray[ImageView]
        An array of all image views
    Returns
    -------
    out : (float,float)
        A tuple with the mean false positive and the mean false negative
    """

    sum_false_positive = 0
    sum_false_negative = 0
    for image_view in image_views.values():
        img_src = project_voxel_centers_on_image(
            voxels_grid.voxels_position,
            voxels_grid.voxels_size,
            image_view.image.shape,
            image_view.projection,
        )

        false_positive, false_negative = image_error(image_view.image, img_src)

        sum_false_positive += false_positive
        sum_false_negative += false_negative

    mean_false_positive = sum_false_positive / len(image_views)
    mean_false_negative = sum_false_negative / len(image_views)

    return mean_false_positive, mean_false_negative



