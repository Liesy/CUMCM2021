from dataprocess import *
from draw_figures import *
from model import *


def main():
    question1_dataprocess()
    question2_dataprocess()

    fig = figures()
    fig.question1_figures()
    fig.question2_1_figures()
    fig.question2_2_figures()

    mod = model()
    mod.model_ols()
    mod.model_regTree2_1_1()
    mod.model_regTree2_1_2()
    mod.model_regTree2_2_1()
    mod.model_regTree2_2_2()
    mod.model_regTree3_1_1()
    mod.model_regTree3_1_2()
    mod.model_regTree3_2_1()
    mod.model_regTree3_2_2()


if __name__ == '__main__':
    main()