from scipy.optimize import curve_fit
from scipy.special import expit
import numpy as np
import pandas as pd



def logistic_5_fitting_no_constraint(x, y):
    def func(x, b0, b1, b2, b3, b4):
        # logistic_part = 0.5 - np.divide(1.0, 1 + np.exp(b1 * (x - b2)))
        logistic_part = 0.5 - expit(b1 * (x - b2))
        y_hat = b0 * logistic_part + b3 * np.asarray(x) + b4
        return y_hat

    x_axis = np.linspace(np.amin(x), np.amax(x), 100)
    init = np.array([np.max(y), np.min(y), np.mean(x), 0.1, 0.1])
    popt, _ = curve_fit(func, x, y, p0=init, maxfev=int(1e8))
    curve = func(x_axis, *popt)
    fitted = func(x, *popt)

    return x_axis, curve, fitted


def IQA_accruacy(y_true, y_pred):
    """
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 准确率
    """
    df=pd.DataFrame({"test_x":y_pred,"test_y":y_true})
    sr=df.corr(method="spearman")
    kd=df.corr(method="kendall")
    
    # _,_,y_pred=logistic_5_fitting_no_constraint(y_pred,y_true)
    df=pd.DataFrame({"test_x":y_pred,"test_y":y_true})
    pl=df.corr(method="pearson")
    rmse=np.sqrt(np.mean(np.square(np.array(y_pred)-np.array(y_true))))
    
    acc=[sr.values[0][1],pl.values[0][1],kd.values[0][1],rmse]
    
    return acc