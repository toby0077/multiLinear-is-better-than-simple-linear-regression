# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:54:54 2018

@author: Administrator
"""

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets, linear_model
from matplotlib.font_manager import FontProperties 
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)  
# additional packages
import sys
import os
sys.path.append(os.path.join('..', '..', 'Utilities'))

try:
# Import formatting commands if directory "Utilities" is available
    from ISP_mystyle import showData 
    
except ImportError:
# Ensure correct performance otherwise
    def showData(*options):
        plt.show()
        return

# additional packages ...
# ... for the 3d plot ...
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ... and for the statistic
from statsmodels.formula.api import ols
#生成组合
from itertools import combinations

x1=[5,2,4,2.5,3,3.5,2.5,3]
x2=[1.5,2,1.5,2.5,3.3,2.3,4.2,2.5]
y=[96,90,95,92,95,94,94,94]


#自变量列表
list_x=[x1,x2]

#绘制多元回归三维图
def Draw_multilinear():
    
    df = pd.DataFrame({'x1':x1,'x2':x2,'y':y})
    # --- >>> START stats <<< ---
    # Fit the model
    model = ols("y~x1+x2", df).fit()
    param_intercept=model.params[0]
    param_x1=model.params[1]
    param_x2=model.params[2]
    rSquared_adj=model.rsquared_adj
    
    #generate data,产生矩阵然后把数值附上去
    x = np.linspace(-5,5,101)
    (X,Y) = np.meshgrid(x,x)
    
    # To get reproducable values, I provide a seed value
    np.random.seed(987654321)   
    Z = param_intercept + param_x1*X+param_x2*Y+np.random.randn(np.shape(X)[0], np.shape(X)[1])

    # 绘图
    #Set the color
    myCmap = cm.GnBu_r
    # If you want a colormap from seaborn use:
    #from matplotlib.colors import ListedColormap
    #myCmap = ListedColormap(sns.color_palette("Blues", 20))
    
    # Plot the figure
    fig = plt.figure("multi")
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,Y,Z, cmap=myCmap, rstride=2, cstride=2, 
        linewidth=0, antialiased=False)
    ax.view_init(20,-120)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("multilinear with adj_Rsquare %f"%(rSquared_adj))
    fig.colorbar(surf, shrink=0.6)
    
    outFile = '3dSurface.png'
    showData(outFile)

    
#检查独立变量之间共线性关系
def Two_dependentVariables_compare(x1,x2):
    # Convert the data into a Pandas DataFrame
    df = pd.DataFrame({'x':x1, 'y':x2})
    # Fit the model
    model = ols("y~x", df).fit()
    rSquared_adj=model.rsquared_adj
    print("rSquared_adj",rSquared_adj)
    if rSquared_adj>=0.8:
        print("high relation")
        return True
    elif 0.6<=rSquared_adj<0.8:
         print("middle relation")
         return False
    elif rSquared_adj<0.6:
         print("low relation")
         return False

#比较所有参数，观察是否存在多重共线
def All_dependentVariables_compare(list_x):  
    list_status=[]
    list_combine=list(combinations(list_x, 2))
    for i in list_combine:
        x1=i[0]
        x2=i[1]
        status=Two_dependentVariables_compare(x1,x2)
        list_status.append(status)
    if True in list_status:
        print("there is multicorrelation exist in dependent variables")
        return True
    else:
        return False
    
        
#回归方程，支持哑铃变量
def regressionModel(x1,x2,y):
    '''Multilinear regression model, calculating fit, P-values, confidence intervals etc.'''
    # Convert the data into a Pandas DataFrame
    df = pd.DataFrame({'x1':x1,'x2':x2,'y':y})
    
    # --- >>> START stats <<< ---
    # Fit the model
    model = ols("y~x1+x2", df).fit()
    # Print the summary
    print((model.summary()))
    return model._results.params  # should be array([-4.99754526,  3.00250049, -0.50514907])

    
# Function to show the resutls of linear fit model
def Draw_linear_line(X_parameters,Y_parameters,figname,x1Name,x2Name):
    #figname表示图表名字，用于生成独立图表fig1 = plt.figure('fig1')，fig2 = plt.figure('fig2')
    plt.figure(figname)
    #获取调整R方参数    
    df = pd.DataFrame({'x':X_parameters, 'y':Y_parameters})
    # Fit the model
    model = ols("y~x", df).fit()
    rSquared_adj=model.rsquared_adj 
    
    #处理X_parameter1数据
    X_parameter1 = []
    for i in X_parameters:
        X_parameter1.append([i])
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameter1, Y_parameters)
    plt.scatter(X_parameter1,Y_parameters,color='blue',label="real value")
    plt.plot(X_parameter1,regr.predict(X_parameter1),color='red',linewidth=4,label="prediction line")
    plt.title("linear regression %s and %s with adj_rSquare:%f"%(x1Name,x2Name,rSquared_adj))
    plt.xlabel('x', fontproperties=font_set)  
    plt.ylabel('y', fontproperties=font_set)  
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()      
    

#绘制多元回归三维图
Draw_multilinear()  
#比较所有参数，观察是否存在多重共线
All_dependentVariables_compare(list_x)              
Draw_linear_line(x1,x2,"fig1","x1","x2")
Draw_linear_line(x1,y,"fig4","x1","y")
Draw_linear_line(x2,y,"fig5","x2","y")
regressionModel(x1,x2,y)
