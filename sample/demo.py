import streamlit as st

from plotter import *
from plotter_definitions import *

st.title('Plotting showroom')

dense_matrix = np.random.rand(100, 100).astype(np.float)
points = np.random.rand(100, 3)

with st.echo():
    spyplot = Spy()
    spyobj = Spy.Object(matrix=dense_matrix > 0.5,
                        axis_visibility='off',
                        markersize=4)
    o = spyplot(spyobj, outfile="sample/spyplot.png")
o

with st.echo():
    imagescplot = Imagesc()
    imagescobj = Imagesc.Object(matrix=dense_matrix,
                                axis_visibility='off',
                                )
    o = imagescplot(imagescobj, outfile="sample/imagescplot.png")
o

with st.echo():
    cloudplot = PlotCloud2D()
    cloudobj = PlotCloud2D.Object(points=points,
                                  axis_visibility='off',
                                  xlim=[0, 1],
                                  ylim=[0, 1],
                                  markersize=500)
    o = cloudplot(cloudobj, outfile="sample/cloudplot.png")
o

with st.echo():
    subplotter = Subplotter()
    o = subplotter(objs=[[spyobj, cloudobj],
                         [imagescobj, spyobj]],
                   plot_functions=[[spyplot, cloudplot],
                                   [imagescplot, spyplot]],
                   outfile="sample/subplot.png",
                   subplot_adjust=0.98)
o

with st.echo():
    animator = Animator()
    for i in range(50):
        dense_matrix = dense_matrix @ dense_matrix
        o = imagescplot(Imagesc.Object(matrix=np.random.rand(100, 100).astype(np.float)))
        animator.add_figure(o)
    animator.save("sample/video.mp4", fps=15)

import io
animator.save("sample/video.gif", fps=15)
f = io.open("sample/video.mp4", 'rb')
st.video(f)
