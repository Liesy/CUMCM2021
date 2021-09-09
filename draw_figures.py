import pandas as pd
import numpy as np
import matplotlib as plt
from dataprocess import *


class figures(object):
    """
    - draw figures from csv
    """

    def __init__(self):
        dataprocess()
        self.df_accessory1 = pd.read_csv('./accessory1.csv', header=0)
        self.df_accessory2 = pd.read_csv('./accessory2.csv', header=0)

    
    def show_figure_1(self):
        pass