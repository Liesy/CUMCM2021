import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataprocess import *


class figures(object):
    """
    - draw figures from csv
    """

    def __init__(self):
        # dataprocess()
        self.df_accessory1 = pd.read_csv('D:/mathematical_modeling/program/accessory1.csv', header=0)
        self.df_accessory2 = pd.read_csv('D:/mathematical_modeling/program/accessory2.csv', header=0)
        self.data_array = np.array(self.df_accessory1)
        self.color_list = ['olive', 'black', 'red', 'springgreen', 'cyan', 'grey', 'darkviolet',
                           'blue', 'aqua', 'cadetblue', 'tomato', 'yellow', 
                           'darkred', 'saddlebrown']
        self.line_style = ['solid', 'dashed']
        

    def question1_figures(self):
        """
        - 温度与其他变量关系的图像
        """
        plt.style.use('ggplot')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示

        # A组
        for num in range(2, 23):
            if num > 15:
                group = 'B' + str(num % 15)
            elif num == 15:
                continue
            else:
                group = 'A' + str(num)
            plt.figure(clear=True)
            plt.title(group)
            for col in range(2, 9):
                plt.plot([self.data_array[row, 1] for row, _ in self.df_accessory1.iterrows() if self.data_array[row, 0] == group],
                         [self.data_array[row, col] for row, _ in self.df_accessory1.iterrows() if self.data_array[row, 0] == group],
                         color=self.color_list[col], label=self.df_accessory1.columns.values[col], linestyle='solid')
            plt.legend()  # 显示图例
            plt.xlabel('温度')
            plt.ylabel('转化率/选择性(%)')
            plt.gcf()
            plt.show()