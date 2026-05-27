# -*- python -*-
#
#       Copyright INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
# ==============================================================================


# ==============================================================================


class ImageView:
    def __init__(self, image, projection):
        self.image = image
        self.projection = projection
        self.integral = None


def iter_image_paths(image_paths, imread, cameras=None, angles=None):

    if cameras is not None:
        cameras = set(cameras)

    if angles is not None:
        angles = set(angles)

    for id_camera in image_paths:

        if cameras is not None and id_camera not in cameras:
            continue

        for angle in image_paths[id_camera]:

            if angles is not None and angle not in angles:
                continue

            yield id_camera, angle, imread(image_paths[id_camera][angle])


def iter_images(images, cameras=None, angles=None):

    if cameras is not None:
        cameras = set(cameras)

    if angles is not None:
        angles = set(angles)

    for id_camera in images:

        if cameras is not None and id_camera not in cameras:
            continue

        for angle in images[id_camera]:

            if angles is not None and angle not in angles:
                continue

            yield id_camera, angle, images[id_camera][angle]


def as_image_views(images_iterator, calibration):
    """Create an ImageView dict from images dict and calibration object

    Args:
        - images_iterator : a (camera_id, angle, image_array) iterator. See iter_images
            or iter_image_paths in openalea.phenomenal.object.
        - calibration: a phenomenal.calibration.Calibration object

    Returns:
        a {f'{camera_id}_{view_angle}': openalea.phenomenal.object.ImageView, ...} dict

    """
    im_views = dict()
    for id_camera, angle, image in images_iterator:
        name = f'{id_camera}_{angle}'
        projection = calibration.get_projection(id_camera, angle)
        im_views[name] = ImageView(
            image,
            projection
        )
    return im_views

