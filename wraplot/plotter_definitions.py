######################################
#  Instances classes
######################################
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes, mtri
from mpl_toolkits.mplot3d import Axes3D

from wraplot import Plotter


class Spy(Plotter):
    @dataclass
    class Object(Plotter.Object):
        matrix: np.ndarray = None  # todo: should't be here https://bugs.python.org/issue36077
        markersize: float = 10

    def __init__(self,
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 figsize: (int, int) = (15, 15)):
        super().__init__(file_dpi, jupy_dpi, figsize)

    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        assert obj.matrix is not None
        ax.spy(obj.matrix, markersize=obj.markersize, zorder=zorder)
        return ax


class Imagesc(Plotter):
    @dataclass
    class Object(Plotter.Object):
        matrix: np.ndarray = None  # todo: should't be here https://bugs.python.org/issue36077

    def __init__(self,
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 figsize: (int, int) = (15, 15)) -> None:
        super().__init__(file_dpi, jupy_dpi, figsize)

    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        assert obj.matrix is not None
        ax.imshow(obj.matrix, zorder=zorder)
        return ax


class PlotCloud2D(Plotter):
    @dataclass
    class Object(Plotter.Object):
        points: np.ndarray = None  # todo: should't be here https://bugs.python.org/issue36077
        xlim: (float, float) = (-1.15, 1.15)
        ylim: (float, float) = (-1.15, 1.15)
        markersize: int = 150
        color: str = '#155084'
        order_color_rgb: np.ndarray = None

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50) -> None:
        super().__init__(file_dpi, jupy_dpi, figsize)

    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        assert obj.points is not None

        color = obj.color
        if obj.order_color_rgb is not None:
            color = np.tile(obj.order_color_rgb, (obj.points.shape[0], 1)) / 255
            color = color * np.linspace(start=0, stop=1, num=obj.points.shape[0])[:, None]
            color[color != color] = 0
        points = np.reshape(obj.points, (-1, obj.points.shape[-1]))
        ax.scatter(points[:, 0], points[:, 1], s=obj.markersize, c=color, marker='.', zorder=zorder)
        return ax


class PlotScalarManifold(Plotter):
    @dataclass
    class Object(Plotter.Object):
        vertices: np.ndarray = None  # todo: should't be here https://bugs.python.org/issue36077
        faces: np.ndarray = None

        manifold_color: Optional[str] = None
        plot_3d: bool = True
        axis_visibility = 'off'

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50) -> None:
        super().__init__(file_dpi, jupy_dpi, figsize)

    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        assert obj.vertices is not None
        assert obj.faces is not None

        vertices = obj.vertices
        faces = obj.faces.astype(int)

        color = obj.manifold_color

        triang = mtri.Triangulation(vertices[:, 0].ravel(),
                                    vertices[:, 1].ravel(), faces)
        ax.view_init(90, 0)
        mm = ax.plot_trisurf(triang, np.zeros((vertices.shape[0], 1)).ravel(),
                             lw=0.4, edgecolor="black", color="white", alpha=1, zorder=zorder)
        colors = np.mean(color[faces], axis=1) if color is not None else vertices[:, 0]
        mm.set_array(colors)
        return ax


class PlotManifold2D(Plotter):
    """
    Plot Manifolds in a standard way
    """

    @dataclass
    class Object(Plotter.Object):
        vertices: np.ndarray = None  # todo: should't be here https://bugs.python.org/issue36077
        faces: np.ndarray = None

        color: (float, float, float) = (202, 62, 71)
        plot_3d: bool = True

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50) -> None:
        super().__init__(file_dpi, jupy_dpi, figsize)

    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        assert obj.vertices is not None
        assert obj.faces is not None

        vertices = obj.vertices
        faces = obj.faces.astype(int)

        ax.triplot(vertices[:, 0], vertices[:, 1], triangles=faces, c=np.asarray(obj.color) / 255, zorder=zorder)
        return ax


class PlotCloudOverManifold2D(Plotter):
    """
    Plot Manifolds in a standard way
    """

    @dataclass
    class Object(Plotter.Object):
        manifold: Any = None  # todo: should't be here https://bugs.python.org/issue36077
        cloud_plotter: Plotter = PlotCloud2D()
        mesh_plotter: Plotter = PlotManifold2D()

    def __init__(self,
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 figsize: (int, int) = (15, 15)):
        """
        Determine how the Manifolds must be plotted

        :param file_dpi: the dpi of the saved image
        :param jupy_dpi: the dpi of the figure
        :param figsize: the dimension of the figure
        """
        super().__init__(file_dpi, jupy_dpi, figsize)

    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        ax = obj.mesh_plotter.plot(ax=ax, obj=obj, zorder=1)
        ax = obj.cloud_plotter.plot(ax=ax, obj=obj, zorder=2)
        return ax


class PlotComparison(Plotter):
    @dataclass
    class Object(Plotter.Object):
        lines: Sequence[np.ndarray] = None  # todo: should't be here https://bugs.python.org/issue36077
        label: Optional[str] = None
        linewidth: float = 1
        legendsize: float = 20
        axis_aspect = 'equal'

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50):
        super().__init__(file_dpi, jupy_dpi, figsize)

    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        assert obj.lines is not None
        lines = obj.lines

        for line in lines:
            ax.plot(line['x'], line['y'],
                    label=obj.label,
                    linewidth=obj.linewidth,
                    zorder=zorder
                    )
        ax.legend(prop={'size': obj.legendsize})
        return ax


class PlotBarsComparison(Plotter):
    @dataclass
    class Object(Plotter.Object):
        bars_value: Sequence[np.ndarray] = None  # todo: should't be here https://bugs.python.org/issue36077
        labels: Sequence[str] = None

        label: Optional[str] = None
        linewidth: float = 1
        legendsize: float = 20
        axis_aspect = 'equal'

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50):
        super().__init__(file_dpi, jupy_dpi, figsize)

    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        assert obj.bars_value is not None
        assert obj.labels is not None

        bars_value = obj.bars_value
        labels = obj.labels

        ax.bar(list(range(len(bars_value))), bars_value, tick_label=labels, zorder=zorder)
        #
        # for tick in ax.xaxis.get_major_ticks():
        #     # specify integer or one of preset strings, e.g.
        #     tick.label.set_fontsize(obj.xla)
        #     # tick.label.set_rotation('vertical')
        return ax


class PlotCoupledBarsComparison(Plotter):
    @dataclass
    class Object(Plotter.Object):
        bar1_values: Sequence[np.ndarray] = None
        bar2_values: Sequence[np.ndarray] = None

        label1: Sequence[str] = None
        label2: Sequence[str] = None

        xticklabels: Sequence[str] = None

        bar_width: float = .35
        linewidth: float = 1
        legendsize: float = 20
        axis_aspect = 'equal'

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50):
        super().__init__(file_dpi, jupy_dpi, figsize)

    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        bar1_values = obj.bar1_values
        bar2_values = obj.bar2_values

        label1 = obj.label1
        label2 = obj.label2

        xticklabels = obj.xticklabels

        width = obj.bar_width

        x = np.arange(len(xticklabels))  # the label locations
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)

        ax.bar(x - width / 2, bar1_values, label=label1, zorder=zorder, width=width)
        ax.bar(x + width / 2, bar2_values, label=label2, zorder=zorder, width=width)

        ax.legend(prop={'size': obj.legendsize})
        return ax


class PlotCloud3D(Plotter):
    @dataclass
    class Object(Plotter.Object):
        points: np.ndarray = None  # todo: should't be here https://bugs.python.org/issue36077

        xlim: (float, float) = (-0.15, 0.15)
        ylim: (float, float) = (-0.15, 0.15)
        zlim: (float, float) = (-0.15, 0.15)

        cmap: str = 'Greens'
        edgecolors: str = 'black'
        linewidths: int = 1
        markersize: int = 1

        plot_3d: bool = True

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 ) -> None:
        super().__init__(file_dpi, jupy_dpi, figsize)

    def plot(self, ax: Axes3D, obj: Object, zorder: int = 1) -> Axes3D:
        assert obj.points is not None

        ax.scatter3D(obj.points[:, 0], obj.points[:, 1], obj.points[:, 2],
                     c=obj.points[:, 2],
                     cmap=obj.cmap,
                     edgecolors=obj.edgecolors,
                     linewidths=obj.linewidths,
                     s=obj.markersize)

        return ax
