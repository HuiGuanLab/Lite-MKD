from collections import namedtuple
import torch
from visdom import Visdom
import numpy as np

Point = namedtuple('point', ['x', 'y'])
PlotInfo = namedtuple('plot_info', ["plot_name", "line_name", "y_label"])

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        
    def plot(self, plot_info:PlotInfo, point:Point):
        x, y = point.x, point.y
        line_name, plot_name, ylabel = plot_info.plot_name, plot_info.line_name, plot_info.y_label
        if ylabel not in self.plots:
            self.plots[ylabel] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[line_name],
                title=plot_name,
                xlabel='Epochs',
                ylabel=ylabel
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[ylabel], name=line_name, update = 'append')
                
            
if __name__ == '__main__':
    visdomLinePlotter = VisdomLinePlotter()
    plotInfo = PlotInfo("test_window", "test_plot", "acc")
    for i in range(10):
        point = Point(x=i, y=i)
        visdomLinePlotter.plot(plotInfo, point)