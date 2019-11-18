from dataclasses import dataclass, field
from typing import Callable, Union, Any, Optional, Sequence, Tuple

import abc
from collections import Iterable
from itertools import cycle
import numpy as np

import matplotlib

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Don't delete
from matplotlib import tri as mtri

import imageio
import collections


######################################
#  Generic classes
######################################

class Plotter(metaclass=abc.ABCMeta):
    @dataclass
    class Object:
        xlim: Optional[Tuple[float, float]] = None
        ylim: Optional[Tuple[float, float]] = None

        title: Optional[str] = field(default=None)
        titlesize: float = field(default=25)

        subtitle: Optional[str] = field(default=None)
        subtitlesize: float = field(default=25)

        xlabel: Optional[str] = field(default=None)
        xlabelsize: float = field(default=25)

        ylabel: Optional[str] = field(default=None)
        ylabelsize: float = field(default=25)

        xticksize: float = field(default=20)
        yticksize: float = field(default=20)

        axis: Optional[str] = field(default=None)
        aspect: str = 'equal'

    def __init__(self,
                 file_dpi: int,
                 jupy_dpi: int,
                 figsize: (int, int),
                 plot_3d: bool = False,
                 zorder: int = 1,
                 axes_equal: bool = True):
        """
        Defines a generic plotter class

        :param file_dpi: the dpi of the saved image
        :param jupy_dpi: the dpi of the figure
        :param figsize: the dimension of the figure
        """
        self.axes_equal = axes_equal
        self.zorder = zorder
        self.figsize = figsize
        self.file_dpi = file_dpi
        self.jupy_dpi = jupy_dpi
        self.plot_3d = plot_3d

    @abc.abstractmethod
    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        pass

    @staticmethod
    def set_static_lims(ax, xlim, ylim):
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

    def get_figure(self, fig, ax, figsize, jupy_dpi, facecolor='w', edgecolor='w'):
        if fig is None or ax is None:
            fig = plt.figure(figsize=figsize, dpi=jupy_dpi, facecolor=facecolor, edgecolor=edgecolor)
            if self.plot_3d:
                ax = fig.add_subplot(111, projection='3d', facecolor=facecolor)
            else:
                ax = fig.gca()
        return fig, ax

    def __call__(self,
                 obj: Object,
                 outfile: str = None,
                 fig: Figure = None,
                 ax: Axes = None,
                 close_fig: bool = False) -> Figure:
        """
        Plot the object and save the results in outfile, if specified.
        It the figure and the axes are given, it reuses those

        :param obj: dictionary that contains the object to plot
        :param outfile: if not None, saves the manifold in the specified path
        :param fig: if given, use this figure instead of creating a new one.
        :param ax:  if given, use this figure instead of creating a new one.
        :param close_fig: if True closes the figure
        :return: the figure
        """
        plt.style.use('ggplot')

        fig, ax = self.get_figure(fig, ax, self.figsize, self.jupy_dpi)

        if self.axes_equal and not self.plot_3d:
            ax.set_aspect('equal')

        self.set_static_lims(ax, obj.xlim, obj.ylim)

        if obj.title:
            fig.suptitle(obj.title, fontsize=obj.titlesize)

        if obj.xlabel:
            ax.set_xlabel(obj.xlabel, fontsize=obj.xlabelsize)

        if obj.ylabel:
            ax.set_ylabel(obj.ylabel, fontsize=obj.ylabelsize)

        if obj.xticksize:
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(obj.xticksize)

        if obj.yticksize:
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(obj.yticksize)

        if obj.axis:
            ax.axis(obj.axis)

        ax = self.plot(ax, obj, self.zorder)

        plt.tight_layout()

        if outfile:
            fig.savefig(fname=outfile, dpi=self.file_dpi)

        if close_fig:
            plt.close(fig)

        return fig


def _flatten(S):
    if isinstance(S, np.ndarray):
        S = S.tolist()

    if not isinstance(S, collections.Sequence):
        return [S]

    if len(S) == 0:
        return S

    if isinstance(S[0], collections.Sequence):
        return _flatten(S[0]) + _flatten(S[1:])

    if len(S[1:]) > 0:
        return S[:1] + _flatten(S[1:])

    return S[:1]


class Subplotter:
    """
    Plot multiple Manifolds in a standard way
    """

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
        self.file_dpi = file_dpi
        self.jupy_dpi = jupy_dpi
        self.figsize = figsize

    def __call__(self,
                 obj: Union[Plotter, Sequence[Plotter.Object], Sequence[Sequence[Plotter.Object]]],
                 plot_functions,
                 # : Optional[
                 #     Union[Callable[[Axes, Plotter.Object], Figure],
                 #           Iterable[Callable[[Axes, Plotter.Object], Figure]]]],
                 subplotadjust: float = 0.9,
                 outfile: str = None,
                 close_fig=False):
        """
        Plot the manifolds and save the results in outfile, if specified.

        :param obj: the objects to plot
        :param outfile: if not None, saves the manifold in the specified path
        :param plot_functions: the function, or sequence of functions, to use to plot each obj
        :param close_fig: if True closes the figure
        :return: the figure
        """
        plt.style.use('ggplot')

        if not isinstance(obj, collections.Sequence):
            num_rows = 1
            num_cols = 1
        elif not isinstance(obj[0], collections.Sequence):
            num_rows = 1
            num_cols = len(obj)
        else:
            num_rows = len(obj)
            num_cols = len(obj[0])

        fig, axes = plt.subplots(num_rows, num_cols, figsize=self.figsize, dpi=self.file_dpi)

        flat_objs = _flatten(obj)
        flat_axes = _flatten(axes)

        plot_functions = cycle(plot_functions) if isinstance(plot_functions, Iterable) else cycle((plot_functions,))

        for (obj_el, ax_el) in zip(flat_objs, flat_axes):
            plot_function = next(plot_functions)
            plot_function(ax=ax_el, obj=obj_el, fig=fig)

            if obj_el.subtitle:
                ax_el.title.set_text(obj_el.subtitle)

        plt.tight_layout()

        plt.subplots_adjust(top=subplotadjust)

        if outfile:
            fig.savefig(fname=outfile, dpi=self.file_dpi)

        if close_fig:
            plt.close(fig)

        return fig


class Animator:
    """
    Generate animations with matplotlib easily
    """

    def __init__(self):
        self.images = []

    def add(self, obj: Plotter.Object, plot_function: Callable[[Axes, Plotter.Object], Figure]) -> np.ndarray:
        """
        Add a frame to the current animation

        :param obj: object to plot
        :param plot_function: the function that knows how to plot the object

        :return: a numpy array representing an image
        """
        fig: Figure = plot_function(obj)
        return self.add_figure(fig)

    def add_figure(self, fig: Figure, close_figure: bool = True) -> np.ndarray:
        """
        Add a frame to the current animation

        :param close_figure: if True closes the figure returned
        :param fig: the figure to add as a frame to the current animation

        :return: a numpy array representing an image
        """
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if close_figure:
            plt.close(fig)
        self.images.append(image)
        return image

    def save(self, filepath: str, fps: int = 30) -> None:
        """
        Save the current animation to file

        :param filepath: the file where to save the animation
        :param fps: the frame-per-second of the animation
        """
        imageio.mimsave(filepath, self.images, fps=fps, subrectangles=True)


######################################
#  Instances classes
######################################


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
        # ax.grid(False)
        return ax


class Imagesc(Plotter):
    @dataclass
    class Object(Plotter.Object):
        matrix: np.ndarray = None  # todo: should't be here https://bugs.python.org/issue36077

    def __init__(self,
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 figsize: (int, int) = (15, 15)
                 ):
        super().__init__(file_dpi, jupy_dpi, figsize)

    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        ax.imshow(obj.matrix, zorder=zorder)
        # ax.grid(False)
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
                 jupy_dpi: int = 50,
                 ):
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


class PlotColormap(Plotter):
    @dataclass
    class Object(Plotter.Object):
        manifold: Any = None  # todo: should't be here https://bugs.python.org/issue36077
        manifold_color: Optional[str] = None

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50):
        super().__init__(file_dpi, jupy_dpi, figsize, plot_3d=True)

    def plot(self, ax: Axes, obj: Object, zorder: int = 1) -> Axes:
        assert obj.manifold is not None
        manifold = obj.manifold
        color = obj.manifold_color

        vertices = manifold.vertices
        faces = manifold.faces.astype(int)

        triang = mtri.Triangulation(vertices[:, 0].ravel(),
                                    vertices[:, 1].ravel(), faces)
        ax.view_init(90, 0)
        plt.axis('off')
        mm = ax.plot_trisurf(triang, np.zeros((vertices.shape[0], 1)).ravel(),
                             lw=0.4, edgecolor="black", color="white", alpha=1, zorder=zorder)
        colors = np.mean(color[faces], axis=1) if color is not None else vertices[:, 0]
        mm.set_array(colors)
        return ax


class PlotManifold(Plotter):
    """
    Plot Manifolds in a standard way
    """

    @dataclass
    class Object(Plotter.Object):
        manifold: Any = None  # todo: should't be here https://bugs.python.org/issue36077
        color: (float, float, float) = (202, 62, 71)

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50):
        super().__init__(file_dpi, jupy_dpi, figsize, plot_3d=True)

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
        assert obj.manifold is not None

        ax.triplot(obj.manifold.vertices[:, 0], obj.manifold.vertices[:, 1], triangles=obj.manifold.faces,
                   c=np.asarray(obj.color) / 255, zorder=zorder)
        return ax


class PlotCloudOverManifold(Plotter):
    """
    Plot Manifolds in a standard way
    """

    @dataclass
    class Object(Plotter.Object):
        manifold: Any = None  # todo: should't be here https://bugs.python.org/issue36077
        cloud_plotter: Plotter = PlotCloud2D()
        mesh_plotter: Plotter = PlotManifold()

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

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50):
        super().__init__(file_dpi, jupy_dpi, figsize, axes_equal=False)

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

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50):
        super().__init__(file_dpi, jupy_dpi, figsize, axes_equal=False)

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

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50):
        super().__init__(file_dpi, jupy_dpi, figsize, axes_equal=False)

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