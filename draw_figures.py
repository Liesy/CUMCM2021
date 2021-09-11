import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataprocess import *


class figures(object):
    """
    - draw figures from csv
    """

    def __init__(self):
        self.df_accessory1 = pd.read_csv('D:/mathematical_modeling/program/accessory1.csv', header=0)
        self.df_accessory2 = pd.read_csv('D:/mathematical_modeling/program/accessory2.csv', header=0)
        self.data_array = np.array(self.df_accessory1)
        self.color_list = ["black", "#2FBF20", "#1F8F9F", "#005FFF", "#905FFF",
                           "#DF00B0", "#FF5020", "#FF8F20", "#FFCF20"]
        self.line_style = ['solid', 'dashed']
        

    def question1_figures(self):
        """
        - 温度与其他变量关系的图像
        """
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示

        # A组
        for num in range(1, 23):
            if num > 15:
                group = 'B' + str(num % 15)
            elif num == 15:
                continue
            else:
                group = 'A' + str(num)
            
            df = self.df_accessory1[(self.df_accessory1['催化剂组合编号'].isin([group]))]
            df.reset_index(drop=True, inplace=True)
            temp = np.array([df.iloc[row, 1] for row,_ in df.iterrows()])
            
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            width = 2
            markersize=15
            linewidth = 1
            # 选择性
            ax1.bar((temp-2*width).tolist(), [df.iloc[row, 3] for row, _ in df.iterrows()], color=self.color_list[3], label=df.columns[3], width=width)
            ax1.bar((temp-width).tolist(), [df.iloc[row, 4] for row, _ in df.iterrows()], color=self.color_list[4], label=df.columns[4], width=width)
            ax1.bar((temp).tolist(), [df.iloc[row, 5] for row, _ in df.iterrows()], color=self.color_list[5], label=df.columns[5], width=width)
            ax1.bar((temp+width).tolist(), [df.iloc[row, 6] for row, _ in df.iterrows()], color=self.color_list[6], label=df.columns[6], width=width)
            ax1.bar((temp+2*width).tolist(), [df.iloc[row, 7] for row, _ in df.iterrows()], color=self.color_list[7], label=df.columns[7], width=width)
            ax1.bar((temp+3*width).tolist(), [df.iloc[row, 8] for row, _ in df.iterrows()], color=self.color_list[8], label=df.columns[8], width=width)
            ax1.legend(bbox_to_anchor=(1.1, 1))
            # 转化率
            ax2.plot(temp, [df.iloc[row, 2] for row, _ in df.iterrows()], 
                     color=self.color_list[2], linestyle='solid', label=df.columns[2],
                     marker='.', markersize=markersize, linewidth=linewidth)
            ax2.legend(bbox_to_anchor=(1.5, 0.5))
            plt.xlabel('温度')
            plt.ylabel('转化率/选择性(%)')
            fig = plt.gcf()
            fig.show()
            saved_path = 'D:/mathematical_modeling/program/figure/' + group + '.png'
            fig.savefig(saved_path, dpi=600, bbox_inches='tight')


    def question2_1_figures(self):
        """
        - 附件二的呈现
        """
        t = np.array([self.df_accessory2.iloc[row, 0] for row,_ in self.df_accessory2.iterrows()])

        plt.style.use('fivethirtyeight')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示

        fig = plt.figure()
        markersize = 15
        linewidth = 1
        for col in range(1, len(self.df_accessory2.columns)):
            plt.plot(t, [self.df_accessory2.iloc[row, col] for row, _ in self.df_accessory2.iterrows()], 
                     color=self.color_list[col], linestyle='solid', label=self.df_accessory2.columns[col],
                     marker='.', markersize=markersize, linewidth=linewidth)
        plt.legend(bbox_to_anchor=(1.05, 0.5))
        plt.xlabel('时间(min)')
        plt.ylabel('转化率/选择性(%)')
        fig = plt.gcf()
        fig.show()
        saved_path = 'D:/mathematical_modeling/program/figure/all.png'
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')


    def question2_2_figures(self):
        """
        - 仅对附件二中乙醇转化率、c4烯烃选择性、c4烯烃收率的呈现
        """
        t = np.array([self.df_accessory2.iloc[row, 0] for row,_ in self.df_accessory2.iterrows()])

        fig = plt.figure()
        markersize = 15
        linewidth = 1
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示
        plt.plot(t, [self.df_accessory2.iloc[row, 1] for row, _ in self.df_accessory2.iterrows()], 
                 color=self.color_list[1], linestyle='solid', label=self.df_accessory2.columns[1],
                 marker='*', markersize=markersize, linewidth=linewidth)
        plt.plot(t, [self.df_accessory2.iloc[row, 3] for row, _ in self.df_accessory2.iterrows()], 
                 color=self.color_list[3], linestyle='solid', label=self.df_accessory2.columns[3],
                 marker='.', markersize=markersize, linewidth=linewidth)
        plt.plot(t, [((self.df_accessory2.iloc[row, 1] * self.df_accessory2.iloc[row, 3]) / 100) for row, _ in self.df_accessory2.iterrows()], 
                 color=self.color_list[len(self.df_accessory2.columns)], linestyle='solid', label='C4烯烃收率(%)',
                 marker='^', markersize=markersize, linewidth=linewidth)
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.xlabel('时间(min)')
        plt.ylabel('转化率/选择性(%)')
        fig = plt.gcf()
        fig.show()
        saved_path = 'D:/mathematical_modeling/program/figure/use.png'
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')