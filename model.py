import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import graphviz
from sklearn.tree import export_graphviz


class model(object):

    def __init__(self):
        self.df_A = pd.read_csv('D:/mathematical_modeling/program/accessoryA.csv', header=0)
        self.df_B = pd.read_csv('D:/mathematical_modeling/program/accessoryB.csv', header=0)
        #生成装填方式的自变量
        self.X = {'A': self.df_A.iloc[:,:5],
                  'B': self.df_B.iloc[:,:5]}        
        #生成装填方式A的因变量
        self.y_A = {'y_1': self.df_A['乙醇转化率(%)'], 
                    'y_2': self.df_A['C4烯烃选择性(%)'], 
                    'y_3': self.df_A['C4烯烃收率(%)']}
        self.y_B = {'y_1': self.df_B['乙醇转化率(%)'], 
                    'y_2': self.df_B['C4烯烃选择性(%)'], 
                    'y_3': self.df_B['C4烯烃收率(%)']}


    def model_ols(self):
        for y in self.y_A:
            model = sm.OLS(self.y_A[y], sm.add_constant(self.df_A.iloc[:,:5])).fit()
            # result = model.fit() #生成模型并拟合
            print(model.summary()) #模型描述

        for y in self.y_B:
            model = sm.OLS(self.y_B[y], sm.add_constant(self.df_B.iloc[:,:5])).fit()
            # result = model.fit() #生成模型并拟合
            print(model.summary()) #模型描述


    def model_regTree2_1_1(self):
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示
        scores = []
        for deep in range(2, 21):
            print('树深:', deep)
            regTree = DecisionTreeRegressor(max_depth=deep, criterion='mse', splitter='best')
            X_train, X_test, y_train, y_test=train_test_split(self.X['A'], self.y_A['y_1'], random_state=0, test_size=0.25)
            regTree.fit(X_train, y_train)
            y_pre = regTree.predict(X_test)
            scores.append(mean_squared_error(y_test, y_pre))
            print('R_squared值:', r2_score(y_test, y_pre))
            print('均方误差:', mean_squared_error(y_test, y_pre))
            print('平均绝对误差:', mean_absolute_error(y_test, y_pre))
        plt.plot(range(2,21), scores, label="testing score")
        plt.legend()
        plt.xlabel('树深')
        plt.ylabel('均方误差')
        fig = plt.gcf()
        fig.show()
        saved_path = 'D:/mathematical_modeling/program/figure/scores2_1_1.png'
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')

        regTree = DecisionTreeRegressor(max_depth=6, criterion='mse', splitter='best')
        regTree.fit(self.X['A'], self.y_A['y_1'])
        dot_data = export_graphviz(regTree, out_file=None)
        graph = graphviz.Source(dot_data)
        # render 方法会在同级目录下生成 Boston PDF文件，内容就是回归树。
        graph.render('./figure/regTree2_1_1')


    def model_regTree2_1_2(self):
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示
        scores = []
        for deep in range(2, 21):
            print('树深:', deep)
            regTree = DecisionTreeRegressor(max_depth=deep, criterion='mse', splitter='best')
            X_train, X_test, y_train, y_test=train_test_split(self.X['A'], self.y_A['y_2'], random_state=0, test_size=0.25)
            regTree.fit(X_train, y_train)
            y_pre = regTree.predict(X_test)
            scores.append(mean_squared_error(y_test, y_pre))
            print('R_squared值:', r2_score(y_test, y_pre))
            print('均方误差:', mean_squared_error(y_test, y_pre))
            print('平均绝对误差:', mean_absolute_error(y_test, y_pre))
        plt.plot(range(2,21), scores, label="testing score")
        plt.legend()
        plt.xlabel('树深')
        plt.ylabel('均方误差')
        fig = plt.gcf()
        fig.show()
        saved_path = 'D:/mathematical_modeling/program/figure/scores2_1_2.png'
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')

        regTree = DecisionTreeRegressor(max_depth=6, criterion='mse', splitter='best')
        regTree.fit(self.X['A'], self.y_A['y_2'])
        dot_data = export_graphviz(regTree, out_file=None)
        graph = graphviz.Source(dot_data)
        # render 方法会在同级目录下生成 Boston PDF文件，内容就是回归树。
        graph.render('./figure/regTree2_1_2')


    def model_regTree2_2_1(self):
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示
        scores = []
        for deep in range(2, 21):
            print('树深:', deep)
            regTree = DecisionTreeRegressor(max_depth=deep, criterion='mse', splitter='best')
            X_train, X_test, y_train, y_test=train_test_split(self.X['B'], self.y_A['y_1'], random_state=0, test_size=0.25)
            regTree.fit(X_train, y_train)
            y_pre = regTree.predict(X_test)
            scores.append(mean_squared_error(y_test, y_pre))
            print('R_squared值:', r2_score(y_test, y_pre))
            print('均方误差:', mean_squared_error(y_test, y_pre))
            print('平均绝对误差:', mean_absolute_error(y_test, y_pre))
        plt.plot(range(2,21), scores, label="testing score")
        plt.legend()
        plt.xlabel('树深')
        plt.ylabel('均方误差')
        fig = plt.gcf()
        fig.show()
        saved_path = 'D:/mathematical_modeling/program/figure/scores2_2_1.png'
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')

        regTree = DecisionTreeRegressor(max_depth=6, criterion='mse', splitter='best')
        regTree.fit(self.X['B'], self.y_A['y_1'])
        dot_data = export_graphviz(regTree, out_file=None)
        graph = graphviz.Source(dot_data)
        # render 方法会在同级目录下生成 Boston PDF文件，内容就是回归树。
        graph.render('./figure/regTree2_2_1')


    def model_regTree2_2_2(self):
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示
        scores = []
        for deep in range(2, 21):
            print('树深:', deep)
            regTree = DecisionTreeRegressor(max_depth=deep, criterion='mse', splitter='best')
            X_train, X_test, y_train, y_test=train_test_split(self.X['B'], self.y_A['y_2'], random_state=0, test_size=0.25)
            regTree.fit(X_train, y_train)
            y_pre = regTree.predict(X_test)
            scores.append(mean_squared_error(y_test, y_pre))
            print('R_squared值:', r2_score(y_test, y_pre))
            print('均方误差:', mean_squared_error(y_test, y_pre))
            print('平均绝对误差:', mean_absolute_error(y_test, y_pre))
        plt.plot(range(2,21), scores, label="testing score")
        plt.legend()
        plt.xlabel('树深')
        plt.ylabel('均方误差')
        fig = plt.gcf()
        fig.show()
        saved_path = 'D:/mathematical_modeling/program/figure/scores2_2_2.png'
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')

        regTree = DecisionTreeRegressor(max_depth=6, criterion='mse', splitter='best')
        regTree.fit(self.X['B'], self.y_A['y_2'])
        dot_data = export_graphviz(regTree, out_file=None)
        graph = graphviz.Source(dot_data)
        # render 方法会在同级目录下生成 Boston PDF文件，内容就是回归树。
        graph.render('./figure/regTree2_2_2')


    def model_regTree3_1_1(self):
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示
        scores = []
        for deep in range(2, 21):
            print('树深:', deep)
            regTree = DecisionTreeRegressor(max_depth=deep, criterion='mse', splitter='best')
            X_train, X_test, y_train, y_test=train_test_split(self.X['A'], self.y_A['y_3'], random_state=0, test_size=0.25)
            regTree.fit(X_train, y_train)
            y_pre = regTree.predict(X_test)
            scores.append(mean_squared_error(y_test, y_pre))
            print('R_squared值:', r2_score(y_test, y_pre))
            print('均方误差:', mean_squared_error(y_test, y_pre))
            print('平均绝对误差:', mean_absolute_error(y_test, y_pre))
        plt.plot(range(2,21), scores, label="testing score")
        plt.legend()
        plt.xlabel('树深')
        plt.ylabel('均方误差')
        fig = plt.gcf()
        fig.show()
        saved_path = 'D:/mathematical_modeling/program/figure/scores3_1_1.png'
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')

        regTree = DecisionTreeRegressor(max_depth=6, criterion='mse', splitter='best')
        regTree.fit(self.X['A'], self.y_A['y_3'])
        dot_data = export_graphviz(regTree, out_file=None)
        graph = graphviz.Source(dot_data)
        # render 方法会在同级目录下生成 Boston PDF文件，内容就是回归树。
        graph.render('./figure/regTree3_1_1')


    def model_regTree3_1_2(self):
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示
        scores = []
        for deep in range(2, 21):
            print('树深:', deep)
            regTree = DecisionTreeRegressor(max_depth=deep, criterion='mse', splitter='best')
            X_train, X_test, y_train, y_test=train_test_split(self.X['B'], self.y_A['y_3'], random_state=0, test_size=0.25)
            regTree.fit(X_train, y_train)
            y_pre = regTree.predict(X_test)
            scores.append(mean_squared_error(y_test, y_pre))
            print('R_squared值:', r2_score(y_test, y_pre))
            print('均方误差:', mean_squared_error(y_test, y_pre))
            print('平均绝对误差:', mean_absolute_error(y_test, y_pre))
        plt.plot(range(2,21), scores, label="testing score")
        plt.legend()
        plt.xlabel('树深')
        plt.ylabel('均方误差')
        fig = plt.gcf()
        fig.show()
        saved_path = 'D:/mathematical_modeling/program/figure/scores3_1_2.png'
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')

        regTree = DecisionTreeRegressor(max_depth=6, criterion='mse', splitter='best')
        regTree.fit(self.X['B'], self.y_A['y_3'])
        dot_data = export_graphviz(regTree, out_file=None)
        graph = graphviz.Source(dot_data)
        # render 方法会在同级目录下生成 Boston PDF文件，内容就是回归树。
        graph.render('./figure/regTree3_1_2')


    def model_regTree3_2_1(self):
        df_A_temp = self.df_A.loc[self.df_A['温度'] < 350]

        X1 = df_A_temp.iloc[:,:5]
        y1 = df_A_temp['C4烯烃收率(%)']

        plt.style.use('fivethirtyeight')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示
        scores = []
        for deep in range(2, 21):
            print('树深:', deep)
            regTree = DecisionTreeRegressor(max_depth=deep, criterion='mse', splitter='best')
            X_train, X_test, y_train, y_test=train_test_split(X1, y1, random_state=0, test_size=0.25)
            regTree.fit(X_train, y_train)
            y_pre = regTree.predict(X_test)
            scores.append(mean_squared_error(y_test, y_pre))
            print('R_squared值:', r2_score(y_test, y_pre))
            print('均方误差:', mean_squared_error(y_test, y_pre))
            print('平均绝对误差:', mean_absolute_error(y_test, y_pre))
        plt.plot(range(2,21), scores, label="testing score")
        plt.legend()
        plt.xlabel('树深')
        plt.ylabel('均方误差')
        fig = plt.gcf()
        fig.show()
        saved_path = 'D:/mathematical_modeling/program/figure/scores3_2_1.png'
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')

        regTree = DecisionTreeRegressor(max_depth=2, criterion='mse', splitter='best')
        regTree.fit(X1, y1)
        dot_data = export_graphviz(regTree, out_file=None)
        graph = graphviz.Source(dot_data)
        # render 方法会在同级目录下生成 Boston PDF文件，内容就是回归树。
        graph.render('./figure/regTree3_2_1')


    def model_regTree3_2_2(self):
        df_B_temp = self.df_B.loc[self.df_B['温度'] < 350]

        X2 = df_B_temp.iloc[:,:5]
        y2 = df_B_temp['C4烯烃收率(%)']

        plt.style.use('fivethirtyeight')
        plt.rcParams['font.sans-serif'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False  # 设置中文编码和符号的正常显示
        scores = []
        for deep in range(2, 21):
            print('树深:', deep)
            regTree = DecisionTreeRegressor(max_depth=deep, criterion='mse', splitter='best')
            X_train, X_test, y_train, y_test=train_test_split(X2, y2, random_state=0, test_size=0.25)
            regTree.fit(X_train, y_train)
            y_pre = regTree.predict(X_test)
            scores.append(mean_squared_error(y_test, y_pre))
            print('R_squared值:', r2_score(y_test, y_pre))
            print('均方误差:', mean_squared_error(y_test, y_pre))
            print('平均绝对误差:', mean_absolute_error(y_test, y_pre))
        plt.plot(range(2,21), scores, label="testing score")
        plt.legend()
        plt.xlabel('树深')
        plt.ylabel('均方误差')
        fig = plt.gcf()
        fig.show()
        saved_path = 'D:/mathematical_modeling/program/figure/scores3_2_2.png'
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')

        regTree = DecisionTreeRegressor(max_depth=2, criterion='mse', splitter='best')
        regTree.fit(X2, y2)
        dot_data = export_graphviz(regTree, out_file=None)
        graph = graphviz.Source(dot_data)
        # render 方法会在同级目录下生成 Boston PDF文件，内容就是回归树。
        graph.render('./figure/regTree3_2_2')