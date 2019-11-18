import typing
from typing import Callable, TypeVar, Union, Dict, Any

import abc
from collections import Iterable
from itertools import cycle
import numpy as np

import matplotlib

# matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Don't delete
from matplotlib import tri as mtri

import imageio


def get(obj, name=None, enforce_presence=False, default_value=None):
    assert obj is not None
    if not isinstance(obj, dict):
        return obj

    assert name is not None
    if name not in obj:
        if enforce_presence:
            raise RuntimeError(f'Missing {name} in dictionary')
        else:
            return default_value

    return obj[name]


######################################
#  Generic classes
######################################

class Plotter(metaclass=abc.ABCMeta):

    def __init__(self,
                 file_dpi: int,
                 jupy_dpi: int,
                 figsize: (int, int),
                 xlim: (float, float),
                 ylim: (float, float),
                 plot_3d: bool = False,
                 zorder: int = 1,
                 axes_equal: bool = True):
        """
        Defines a generic plotter class

        :param file_dpi: the dpi of the saved image
        :param jupy_dpi: the dpi of the figure
        :param figsize: the dimension of the figure
        :param ylim: the limits for the y-axis
        :param xlim: the limits for the x-axis
        """
        self.axes_equal = axes_equal
        self.zorder = zorder
        self.figsize = figsize
        self.file_dpi = file_dpi
        self.jupy_dpi = jupy_dpi
        self.ylim = ylim
        self.xlim = xlim
        self.plot_3d = plot_3d

    @abc.abstractmethod
    def plot(self, ax: Axes, obj: Dict[str, Any], zorder: int = 1) -> Axes:
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
                 obj: Dict[str, Any],
                 outfile: str = None,
                 fig: Figure = None,
                 ax: Axes = None,
                 close_fig: bool = False,
                 xlim: (float, float) = None,
                 ylim: (float, float) = None,
                 title: str = None,  # todo: deprecated. Use obj API for display config
                 ) -> Figure:
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

        self.set_static_lims(ax,
                             xlim if xlim else self.xlim,
                             ylim if ylim else self.ylim)

        if title:
            plt.title(title)

        if 'title' in obj:
            plt.title(get(obj, 'title'), fontsize=get(obj, 'titlesize', default_value=25))

        if 'xlabel' in obj:
            ax.set_xlabel(get(obj, 'xlabel'), fontsize=get(obj, 'xlabelsize', default_value=25))

        if 'ylabel' in obj:
            ax.set_ylabel(get(obj, 'ylabel'), fontsize=get(obj, 'ylabelsize', default_value=25))

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(get(obj, 'xticksize', default_value=20))

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(get(obj, 'yticksize', default_value=20))

        if 'axis' in obj:
            ax.axis(get(obj, 'axis'))

        ax = self.plot(ax, obj, self.zorder)

        plt.tight_layout()

        if outfile:
            fig.savefig(fname=outfile, dpi=self.file_dpi)

        if close_fig:
            plt.close(fig)

        return fig


import collections


def flatten(S):
    if isinstance(S, np.ndarray):
        S = S.tolist()
        
    if not isinstance(S, collections.Sequence):
        return [S]
    
    if len(S) == 0:
        return S
    
    if isinstance(S[0], collections.Sequence):
        return flatten(S[0]) + flatten(S[1:])
    
    if len(S[1:])>0:
        return S[:1] + flatten(S[1:])

    return S[:1]

class Subplotter:
    """
    Plot multiple Manifolds in a standard way
    """
    T = TypeVar('T')

    def __init__(self,
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 figsize: (int, int) = (15, 15)):
        """
        Determine how the Manifolds must be plotted

        :param num_rows: number of rows in the figure
        :param num_cols: number of cols in the figure
        :param file_dpi: the dpi of the saved image
        :param jupy_dpi: the dpi of the figure
        :param figsize: the dimension of the figure
        """
        self.file_dpi = file_dpi
        self.jupy_dpi = jupy_dpi
        self.figsize = figsize

    def __call__(self,
                 obj: typing.Sequence[T],
                 plot_functions: typing.Optional[
                     Union[Callable[[T, Figure, Axes], Figure],
                           typing.Iterable[Callable[[T, Figure, Axes], Figure]]]] = None,
                 outfile: str = None,
                 close_fig=False,
                 xlim: (float, float) = None,
                 ylim: (float, float) = None,
                 ):
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

        flat_objs = flatten(obj)
        flat_axes = flatten(axes)

        plot_functions = cycle(plot_functions) if isinstance(plot_functions, Iterable) else cycle((plot_functions,))

        for (obj_el, ax_el) in zip(flat_objs, flat_axes):
            plot_function = next(plot_functions)
            plot_function(ax=ax_el, obj=obj_el, fig=fig,  xlim=xlim, ylim=ylim, )

        plt.tight_layout()

        if outfile:
            fig.savefig(fname=outfile, dpi=self.file_dpi)

        if close_fig:
            plt.close(fig)

        return fig


class Animator:
    """
    Generate animations with matplotlib easily
    """

    T = TypeVar('T')

    def __init__(self):
        self.images = []

    def add(self, obj: T, plot_function: Callable[[T], Figure]) -> np.ndarray:
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
#  Subplot Utils
######################################

# class ManifoldSubplots:
#     """
#     Plot multiple Manifolds in a standard way
#     """

#     def __init__(self,
#                  num_rows: int = 2,
#                  num_cols: int = 1,
#                  file_dpi: int = 150,
#                  jupy_dpi: int = 50,
#                  figsize: (int, int) = (15, 15)):
#         """
#         Determine how the Manifolds must be plotted

#         :param num_rows: number of rows in the figure
#         :param num_cols: number of cols in the figure
#         :param file_dpi: the dpi of the saved image
#         :param jupy_dpi: the dpi of the figure
#         :param figsize: the dimension of the figure
#         """
#         self.default_plotter = PlotManifold()
#         self.subplotter = Subplotter(num_rows=num_rows, num_cols=num_cols, file_dpi=file_dpi,
#                                      jupy_dpi=jupy_dpi, figsize=figsize)

#     def __call__(self,
#                  manifolds: Union[typing.Iterable[Manifold], Dict[str, np.ndarray]],
#                  outfile: str = None,
#                  plot_function: Callable[[Any], Figure] = None,
#                  close_fig=False,
#                  xlim: (float, float) = None,
#                  ylim: (float, float) = None, ):
#         """
#         Plot the manifolds and save the results in outfile, if specified.

#         :param manifolds: the manifolds to plot
#         :param outfile: if not None, saves the manifold in the specified path
#         :param plot_function: the function to use to plot each manifold
#         :param close_fig: if True closes the figure
#         :return: the figure
#         """
#         if isinstance(manifolds, dict):
#             manifolds = [Manifold(manifolds['vertices'][i], manifolds['faces'][i])
#                          for i in range(manifolds['vertices'].shape[0])]
#         plotter = plot_function if plot_function else self.default_plotter

#         return self.subplotter(manifolds, plot_functions=plotter, outfile=outfile, close_fig=close_fig, xlim=xlim,
#                                ylim=ylim)


######################################
#  Instances classes
######################################


class Spy(Plotter):
    def __init__(self,
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 figsize: (int, int) = (15, 15),
                 xlim: (float, float) = None,
                 ylim: (float, float) = None,
                 markersize=0.5, ):
        super().__init__(file_dpi, jupy_dpi, figsize, xlim, ylim)
        self.markersize = markersize

    def plot(self, ax: Axes, obj: Dict[str, Any], zorder: int = 1) -> Axes:
        matrix = get(obj, 'matrix', enforce_presence=True)
        markersize = get(obj, 'markersize', default_value=self.markersize)
        ax.spy(matrix, markersize=markersize, zorder=zorder)
        # ax.grid(False)
        return ax


class Imagesc(Plotter):
    def __init__(self,
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 figsize: (int, int) = (15, 15),
                 xlim: (float, float) = None,
                 ylim: (float, float) = None):
        super().__init__(file_dpi, jupy_dpi, figsize, xlim, ylim)

    def plot(self, ax: Axes, obj: Dict[str, Any], zorder: int = 1) -> Axes:
        matrix = get(obj, 'matrix', enforce_presence=True)
        ax.imshow(matrix, zorder=zorder)
        # ax.grid(False)
        return ax


class PlotCloud2D(Plotter):

    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 xlim: (float, float) = (-1.15, 1.15),
                 ylim: (float, float) = (-1.15, 1.15)):
        super().__init__(file_dpi, jupy_dpi, figsize, xlim, ylim)
        self.markersize = 150
        self.color = '#155084'

    def plot(self, ax: Axes, obj: Dict[str, Any], zorder: int = 1) -> Axes:
        points = get(obj, 'points', enforce_presence=True)
        markersize = get(obj, 'markersize', default_value=self.markersize)
        color = self.color

        if 'order_color_rgb' in obj:
            color = np.tile(obj['order_color_rgb'], (points.shape[0], 1)) / 255
            color = color * np.linspace(start=0, stop=1, num=points.shape[0])[:, None]
            color[color != color] = 0
        points = np.reshape(points, (-1, points.shape[-1]))
        ax.scatter(points[:, 0], points[:, 1], s=markersize, c=color, marker='.', zorder=zorder)
        return ax


class PlotColormap(Plotter):
    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 xlim: (float, float) = None,
                 ylim: (float, float) = None):
        super().__init__(file_dpi, jupy_dpi, figsize, xlim, ylim, plot_3d=True)

    def plot(self, ax: Axes, obj: Dict[str, Any], zorder: int = 1) -> Axes:
        manifold = get(obj, 'manifold', enforce_presence=True)
        color = get(obj, 'manifold_color', enforce_presence=False)

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

    def __init__(self,
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 figsize: (int, int) = (15, 15),
                 xlim: (float, float) = (-1.15, 1.15),
                 ylim: (float, float) = (-1.15, 1.15)):
        """
        Determine how the Manifolds must be plotted

        :param color: the color of the manifold
        :param file_dpi: the dpi of the saved image
        :param jupy_dpi: the dpi of the figure
        :param figsize: the dimension of the figure
        :param ylim: the limits for the y-axis
        :param xlim: the limits for the x-axis
        """
        super().__init__(file_dpi, jupy_dpi, figsize, xlim, ylim)
        self.color = (202, 62, 71)

    def plot(self, ax: Axes, obj: Dict[str, Any], zorder: int = 1) -> Axes:
        manifold = get(obj, 'manifold', enforce_presence=True)
        color = get(obj, 'manifold_edge_color', default_value=self.color)
        

        ax.triplot(manifold.vertices[:, 0], manifold.vertices[:, 1], triangles=manifold.faces,
                   c=np.asarray(color) / 255, zorder=zorder)
        return ax


class PlotCloudOverManifold(Plotter):
    """
    Plot Manifolds in a standard way
    """

    def __init__(self,
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 figsize: (int, int) = (15, 15),
                 xlim: (float, float) = (-1.15, 1.15),
                 ylim: (float, float) = (-1.15, 1.15)):
        """
        Determine how the Manifolds must be plotted

        :param color: the color of the manifold
        :param file_dpi: the dpi of the saved image
        :param jupy_dpi: the dpi of the figure
        :param figsize: the dimension of the figure
        :param ylim: the limits for the y-axis
        :param xlim: the limits for the x-axis
        """
        super().__init__(file_dpi, jupy_dpi, figsize, xlim, ylim)
        self.cloud_plotter = PlotCloud2D(file_dpi, jupy_dpi, figsize, xlim, ylim)
        self.mesh_plotter = PlotManifold(file_dpi, jupy_dpi, figsize, xlim, ylim)

    def plot(self, ax: Axes, obj: Dict[str, Any], zorder: int = 1) -> Axes:
        ax = self.mesh_plotter.plot(ax=ax, obj=obj, zorder=1)
        ax = self.cloud_plotter.plot(ax=ax, obj=obj, zorder=2)
        return ax


class PlotComparison(Plotter):
    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 xlim: (float, float) = None,
                 ylim: (float, float) = None, ):
        super().__init__(file_dpi, jupy_dpi, figsize, xlim, ylim, plot_3d=False, axes_equal=False)

    def plot(self, ax: Axes, obj: Dict[str, Any], zorder: int = 1) -> Axes:
        lines = get(obj, 'lines', enforce_presence=True)
        for line in lines:
            ax.plot(line['x'], line['y'],
                    label=line['label'] if 'label' in line else None,
                    linewidth=get(obj, 'linewidth', default_value= 1),
                    zorder=zorder
                    )
        ax.legend(prop={'size': get(obj, 'legendsize', default_value= 20)})
        return ax


class PlotBarsComparison(Plotter):
    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 xlim: (float, float) = None,
                 ylim: (float, float) = None, ):
        super().__init__(file_dpi, jupy_dpi, figsize, xlim, ylim, plot_3d=False, axes_equal=False)

    def plot(self, ax: Axes, obj: Dict[str, Any], zorder: int = 1) -> Axes:
        bars_value = get(obj, 'bars', enforce_presence=True)
        labels = get(obj, 'labels', enforce_presence=True)

        ax.bar(list(range(len(bars_value))), bars_value, tick_label=labels, zorder=zorder)

        for tick in ax.xaxis.get_major_ticks():
            # specify integer or one of preset strings, e.g.
            tick.label.set_fontsize(25)
            # tick.label.set_rotation('vertical')
        return ax


class PlotCoupledBarsComparison(Plotter):
    def __init__(self,
                 figsize: (int, int) = (15, 15),
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 xlim: (float, float) = None,
                 ylim: (float, float) = None, ):
        super().__init__(file_dpi, jupy_dpi, figsize, xlim, ylim, plot_3d=False, axes_equal=False)

    def plot(self, ax: Axes, obj: Dict[str, Any], zorder: int = 1) -> Axes:
        bar1_values = get(obj, 'bar1_values', enforce_presence=True)
        bar2_values = get(obj, 'bar2_values', enforce_presence=True)
        
        label1 = get(obj, 'label1', enforce_presence=True)
        label2 = get(obj, 'label2', enforce_presence=True)
        
        xticklabels = get(obj, 'xticklabels', enforce_presence=True)
        
        
        width = get(obj, 'width', enforce_presence=False, default_value=0.35)

        x = np.arange(len(xticklabels))  # the label locations
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)

        ax.bar(x - width/2, bar1_values, label=label1, zorder=zorder, width=width)
        ax.bar(x + width/2, bar2_values, label=label2, zorder=zorder, width=width)

        ax.legend(prop={'size': get(obj, 'legendsize', default_value= 20)})
        return ax
    