# -*- python -*-
#
#       Copyright INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
# ==============================================================================


import numpy
import k3d
import ipyvolume
from ._order_color_map import order_color_map
# ==============================================================================


def plot_voxel(voxels_position, color=0x00ff00, size=2.0):
    plt_points = k3d.points(positions=voxels_position.astype(numpy.float32),
                            point_size=size, color=color)
    return plt_points


def show_point_cloud(xyz_positions,
                     color=0x00ff00,
                     size=2):

    plot = k3d.plot()
    plot += plot_voxel(xyz_positions, color, size)
    plot.display()

def show_voxel_grid(voxel_grid,
                    color=0x00ff00,
                    size=2):

    plot = k3d.plot()
    plot += plot_voxel(voxel_grid.voxels_position, size=size, color=color)
    plot.display()


def show_mesh(vertices, faces, color=0x00ff00):
    mesh = k3d.mesh(vertices.astype(numpy.float32), indices=faces.astype(numpy.uint32), color=color)
    plot = k3d.plot()
    plot += mesh
    return plot

def show_skeleton(
    voxel_skeleton,
    size=2,
    with_voxel=True,
    voxels_color=0x00ff00,
    polyline_color=0xff0000,
):
    plot = k3d.plot()
    if with_voxel:
        voxels_position = voxel_skeleton.voxels_position()
        plot += plot_voxel(voxels_position, size=size / 2, color=voxels_color)

    voxels_position = voxel_skeleton.voxels_position_polyline()
    plot += plot_voxel(voxels_position, size=size, color=polyline_color)

    for vs in voxel_skeleton.segments:
        for color, index in [(0x0000ff, 0), (0xff0000, -1)]:
            plot += plot_voxel(
                numpy.array([vs.polyline[index]]),
                size=size * 2,
                color=color,
            )
    plot.display()


def show_segmentation(voxel_segmentation, size=2.0, width=500, height=500):
    ipyvolume.figure(width=width, height=height)
    ipyvolume.view(180, 90)

    def get_color(label, info):
        if label == "stem":
            color = (128, 128, 128)
        elif label == "unknown":
            color = (255, 255, 255)
        elif "pm_leaf_number" in info:
            color_map = order_color_map()
            color = color_map[info["pm_leaf_number"]]
            color = tuple([int(255 * x) for x in color])
        else:
            if label == "growing_leaf":
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

        return "rgb" + str(color)

    for vo in voxel_segmentation.voxel_organs:
        voxels_position = numpy.array(list(map(tuple, list(vo.voxels_position()))))

        plot_voxel(voxels_position, size=size * 1, color=get_color(vo.label, vo.info))

        if (
            (vo.label == "mature_leaf" or vo.label == "growing_leaf")
            and len(vo.voxel_segments) > 0
            and "pm_position_tip" in vo.info
        ):
            plot_voxel(
                numpy.array([vo.info["pm_position_tip"]]),
                size=size * 2,
                color="red",
                marker="sphere",
            )

            plot_voxel(
                numpy.array([vo.info["pm_position_base"]]),
                size=size * 2,
                color="blue",
                marker="sphere",
            )

    voxels_position = numpy.array(list(voxel_segmentation.get_voxels_position()))

    x_min = voxels_position[:, 0].min()
    x_max = voxels_position[:, 0].max()
    y_min = voxels_position[:, 1].min()
    y_max = voxels_position[:, 1].max()
    z_min = voxels_position[:, 2].min()
    z_max = voxels_position[:, 2].max()
    xyz_max = max(x_max - x_min, y_max - y_min, z_max - z_min)
    ipyvolume.xlim(x_min, x_min + xyz_max)
    ipyvolume.ylim(y_min, y_min + xyz_max)
    ipyvolume.zlim(z_min, z_min + xyz_max)
    ipyvolume.show()


def show_synthetic_plant(
    vertices, faces, meta_data=None, size=0.5, color="green", width=500, height=500
):
    ipyvolume.figure(width=width, height=height)
    ipyvolume.view(180, 90)

    ipyvolume.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color=color
    )

    voxels_position = vertices
    if meta_data is not None:
        ranks = meta_data["leaf_order"]
        polylines = {
            n: list(map(numpy.array, list(zip(*meta_data["leaf_polylines"][i]))))
            for i, n in enumerate(ranks)
        }

        voxels = set()
        for leaf_order in polylines:
            x, y, z, r = polylines[leaf_order]
            polyline = numpy.array(list(zip(x, y, z))) * 10 - numpy.array([0, 0, 750])

            plot_voxel(polyline, size=size, color="red")
            voxels = voxels.union(set(map(tuple, list(polyline))))

        voxels = voxels.union(set(map(tuple, list(voxels_position))))
        voxels_position = numpy.array(list(voxels), dtype=numpy.intp)

    x_min = voxels_position[:, 0].min()
    x_max = voxels_position[:, 0].max()
    y_min = voxels_position[:, 1].min()
    y_max = voxels_position[:, 1].max()
    z_min = voxels_position[:, 2].min()
    z_max = voxels_position[:, 2].max()
    xyz_max = max(x_max - x_min, y_max - y_min, z_max - z_min)
    ipyvolume.xlim(x_min, x_min + xyz_max)
    ipyvolume.ylim(y_min, y_min + xyz_max)
    ipyvolume.zlim(z_min, z_min + xyz_max)

    ipyvolume.show()
