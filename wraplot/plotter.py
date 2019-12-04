import os
from dataclasses import dataclass, field
from typing import Union, Optional, Sequence, Tuple, TypeVar, Iterable

import abc
import collections
from itertools import cycle
import numpy as np

import matplotlib

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Don't delete
from matplotlib import tri as mtri

import imageio


######################################
#  Generic classes
######################################
class Plotter(metaclass=abc.ABCMeta):
    @dataclass
    class Object:
        # General appearance
        matplotlib_style: str = "ggplot"

        # Axis
        plot_3d: bool = False
        axis: Optional[str] = field(default=None)
        axis_aspect: str = 'equal'
        axis_grid: Optional[bool] = None
        axis_visibility: Optional[str] = None
        axis_view_init: Optional[Tuple[float, float]] = None

        xlim: Optional[Tuple[float, float]] = None
        ylim: Optional[Tuple[float, float]] = None
        zlim: Optional[Tuple[float, float]] = None

        # Sizes
        titlesize: float = field(default=25)
        subtitlesize: float = field(default=25)
        xlabelsize: float = field(default=25)
        ylabelsize: float = field(default=25)

        xticksize: float = field(default=20)
        yticksize: float = field(default=20)

        # Labels
        title: Optional[str] = field(default=None)
        subtitle: Optional[str] = field(default=None)
        xlabel: Optional[str] = field(default=None)
        ylabel: Optional[str] = field(default=None)

    def __init__(self,
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 figsize: (int, int) = (15, 15),
                 zorder: int = 1) -> None:
        """
        Defines a generic plotter class

        :param file_dpi: the dpi of the saved image
        :param jupy_dpi: the dpi of the figure
        :param figsize: the dimension of the figure
        """
        self.figsize: (int, int) = figsize
        self.file_dpi: int = file_dpi
        self.jupy_dpi: int = jupy_dpi
        self.zorder = zorder

    @abc.abstractmethod
    def plot(self,
             ax: Union[Axes3D, Axes],
             obj: Object,
             zorder: int = 1) -> Union[Axes3D, Axes]:
        pass

    @staticmethod
    def set_static_lims(ax: Union[Axes3D, Axes],
                        xlim: Tuple[float, float],
                        ylim: Tuple[float, float],
                        zlim: Tuple[float, float]) -> None:
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        if zlim:
            ax.set_zlim(*zlim)

    @staticmethod
    def get_figure(fig: Figure,
                   ax: Union[Axes3D, Axes],
                   plot_3d: bool,
                   figsize: (int, int),
                   jupy_dpi: int,
                   facecolor: Union[str, Tuple[float, float, float]] = 'w',
                   edgecolor: Union[str, Tuple[float, float, float]] = 'w') -> (Figure, Union[Axes3D, Axes]):
        if fig is None or ax is None:
            fig = plt.figure(figsize=figsize, dpi=jupy_dpi, facecolor=facecolor, edgecolor=edgecolor)
            if plot_3d:
                ax = fig.add_subplot(111, projection='3d', facecolor=facecolor)
            else:
                ax = fig.gca()
        return fig, ax

    @staticmethod
    def extract_image(fig: Figure, close_figure: bool = True) -> np.ndarray:
        """
        Extract a np.ndarray representation of the Figure

        :param fig: the Figure to transform in image
        :param close_figure: if True, closes the Figure
        """
        # todo: add support to alpha channel somehow
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if close_figure:
            plt.close(fig)
        return image

    def __call__(self,
                 obj: Object,
                 outfile: str = None,
                 fig: Figure = None,
                 ax: Union[Axes3D, Axes] = None,
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

        assert isinstance(obj, Plotter.Object), f"{type(obj)} is not are plottable. It should be {Plotter.Object}"

        plt.style.use(obj.matplotlib_style)

        fig, ax = self.get_figure(fig, ax, obj.plot_3d, self.figsize, self.jupy_dpi)

        if obj.axis_aspect and not (obj.axis_aspect == 'equal' and obj.plot_3d):
            ax.set_aspect(obj.axis_aspect)

        self.set_static_lims(ax, obj.xlim, obj.ylim, obj.zlim)

        if obj.axis_grid is not None:
            ax.grid(obj.axis_grid)

        if obj.axis_visibility is not None:
            if obj.axis_visibility == 'off':
                ax.set_axis_off()

        if obj.axis_view_init is not None:
            ax.view_init(*obj.axis_view_init)

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


T = TypeVar('T')


def _flatten(to_flatten: Union[Sequence[T], np.ndarray]) -> Sequence[T]:
    """
    Flatten nested sequences to a flat list
    """
    if isinstance(to_flatten, np.ndarray):
        to_flatten = to_flatten.tolist()

    if not isinstance(to_flatten, collections.abc.Sequence):
        return [to_flatten]

    if len(to_flatten) == 0:
        return to_flatten

    if isinstance(to_flatten[0], collections.abc.Sequence):
        # todo: Solve type hint warning
        return _flatten(to_flatten[0]) + _flatten(to_flatten[1:])

    if len(to_flatten[1:]) > 0:
        return to_flatten[:1] + _flatten(to_flatten[1:])

    return to_flatten[:1]


class Subplotter:
    """
    Plot multiple Plotter.Objects in the same Figure
    """

    def __init__(self,
                 file_dpi: int = 150,
                 jupy_dpi: int = 50,
                 figsize: (int, int) = (15, 15)) -> None:
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
                 objs: Union[Plotter.Object, Sequence[Plotter.Object], Sequence[Sequence[Plotter.Object]]],
                 plot_functions: Union[Plotter, Sequence[Plotter], Sequence[Sequence[Plotter]]],
                 subplot_adjust: float = 0.9,
                 outfile: str = None,
                 close_fig: bool = False) -> Figure:
        """
        Plot the manifolds and save the results in outfile, if specified.

        :param objs: the objects to plot
        :param outfile: if not None, saves the manifold in the specified path
        :param plot_functions: the function, or sequence of functions, to use to plot each obj
        :param subplot_adjust: adjust the distance between the title and the subplots
        :param outfile: if not None, save to the specified file
        :param close_fig: if True closes the figure
        :return: the figure
        """

        if not isinstance(objs, collections.abc.Sequence):
            num_rows = 1
            num_cols = 1
        elif not isinstance(objs[0], collections.abc.Sequence):
            num_rows = 1
            num_cols = len(objs)
        else:
            num_rows = len(objs)
            num_cols = len(objs[0])

        fig, axes = plt.subplots(num_rows, num_cols, figsize=self.figsize, dpi=self.file_dpi)

        flat_objs = _flatten(objs)
        flat_axes = _flatten(axes)
        flat_plot = _flatten(plot_functions)

        plt.style.use(flat_objs[0].matplotlib_style)

        assert len(flat_axes) == len(
            flat_objs) == num_cols * num_rows, f'Got {len(flat_objs)} objects, expected: {len(flat_axes)}'

        plot_functions = cycle(flat_plot)

        for (obj_el, ax_el, plot_function) in zip(flat_objs, flat_axes, flat_plot):
            plot_function(obj=obj_el, ax=ax_el, fig=fig)

            if obj_el.subtitle:
                ax_el.title.set_text(obj_el.subtitle)

        plt.tight_layout()

        plt.subplots_adjust(top=subplot_adjust)

        if outfile:
            fig.savefig(fname=outfile, dpi=self.file_dpi)

        if close_fig:
            plt.close(fig)

        return fig


class Animator:
    """
    Generate animations with matplotlib and imageio
    """

    def __init__(self) -> None:
        self.images = []

    def add(self, obj: Plotter.Object, plot_function: Plotter) -> np.ndarray:
        """
        Add a frame to the current animation

        :param obj: object to plot
        :param plot_function: the function that knows how to plot the object

        :return: a numpy array representing an image
        """
        fig: Figure = plot_function(obj)
        return self.add_figure(fig, close_figure=True)

    def add_figure(self, fig: Figure, close_figure: bool = True) -> np.ndarray:
        """
        Add a frame to the current animation

        :param close_figure: if True closes the figure returned
        :param fig: the figure to add as a frame to the current animation

        :return: a numpy array representing an image
        """
        image = Plotter.extract_image(fig, close_figure)
        self.images.append(image)
        return image

    def save(self, filepath: str, anim_format: str = '.mp4', fps: int = 30) -> None:
        """
        Save the current animation to file

        :param filepath: the file where to save the animation
        :param anim_format: the format of the output animation
        :param fps: the frame-per-second of the animation
        """
        name, ext = os.path.splitext(filepath)
        if not ext:
            ext = anim_format
        filepath = f'{name}{ext}'

        if ext == ".gif":
            return imageio.mimsave(filepath, self.images, fps=fps, subrectangles=True)
        else:
            return imageio.mimsave(filepath, self.images, fps=fps)
