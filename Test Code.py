#!/usr/bin/env python
# coding: utf-8

# In[14]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tsfresh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sklearn as sk
import pickle
import pywt

def main_method(abc):
    x=[]
    linear_trend=[]
    ent=[]
    bent=[]
    se=[]
    lstr=[]
    

    csvfileClean=series_data_clean(abc)
    for i,row in csvfileClean.iterrows() :
            #print(row)
            x=np.array(row)
            x=x.flatten()
         
            ent.append(tsfresh.feature_extraction.feature_calculators.sample_entropy(x))
            bent.append(tsfresh.feature_extraction.feature_calculators.binned_entropy(x, len(x)))
            

    return ent,bent

#Feature 2    
def get_velocities(abc):

    df=series_data_clean(abc)
 
    max_cgm_velocity=[]
    min_cgm_velocity=[]
    zero_crossing=[]
    max_indices=[]
    min_indices=[]
    data_list=df.values.tolist()

    x=len(data_list)
    y=len(data_list[0])
    time_diff=5
    for i in range(0,x):
        max_vel=-1000
        min_vel=1000000
        count=0
        max_index=0
        min_index=0
        row=data_list[0]
        zero_crossing.append(tsfresh.feature_extraction.feature_calculators.number_crossing_m(row, 0))
        for j in range(y-1,0,-1):

            cgm_diff=abs(data_list[i][j]-data_list[i][j-1])
            cgm_vel=cgm_diff/time_diff
            if cgm_vel>max_vel:
                max_vel=cgm_vel
                max_index=j
            if cgm_vel<min_vel:
                min_vel=cgm_vel
                min_index=j
            if cgm_diff == 0:
                count=count+1


#         print(max_index)
        zero_crossing.append(count)
        max_cgm_velocity.append(max_vel)
        min_cgm_velocity.append(min_vel)
        max_indices.append(max_index)
        min_indices.append(min_index)
    return max_cgm_velocity, min_cgm_velocity
#Feature 3
def feature_DWT(abc):
    data=series_data_clean(abc)
    dwt_frame=pd.DataFrame()
    for index, row in data.iterrows():
        approx, coeff=pywt.dwt(row,'db1')
        coeff=pd.Series(coeff)
        dwt_frame =   dwt_frame.append(coeff, ignore_index=True)

    column=[]
    for i in range(0, len(coeff)):
        column.append('dwt_coef'+str(i))
    dwt_frame.columns = column

    return dwt_frame

def DWT_plot(value):
    ax = plt.gca()
#     print(value)

    for index in range(value.shape[1]):
        columnSeriesObj = value.iloc[: , index]
        a=np.arange(0, len(columnSeriesObj))
#         plt.scatter(a, columnSeriesObj, 1)
        plt.plot(a, columnSeriesObj, 1)

def series_data_clean(x):
    s_data=pd.read_csv(x)
    temp=s_data.interpolate(method='quadratic')
    temp_s=temp.interpolate(method='linear')
    
    new_s=temp_s.dropna()
#     print(new_s)
#     print(new_s["cgmSeries_ 3"])
    return new_s

def date_time_data_clean():
    t_data=pd.read_csv('CGMDatenumLunchPat4.csv')
    temp2=t_data.interpolate(method='quadratic')
    temp_d=temp2.interpolate(method='linear')
    new_d=temp_d.dropna()
    return new_d

#Feature 4
def polyfit_feature(abc):
    
    csvfileClean3=series_data_clean(abc)
    poly_coefficients =[]
    #poly_coefficients is a list that stores the values returned by the polyfit funtion
    for row in csvfileClean3.iterrows() :

        y=np.array(row[1])
        y= y.flatten()
        x= np.arange(1,(len(y)+1),1)
        poly_coefficients.append(np.polyfit(np.float32(x),y, deg=5))

    w1 = []
    w2 = []
    w3 = []
    w4 = []
    w5 = []
    weights = [w5,w4,w3,w2,w1]
    for i in range (len(poly_coefficients)):
        # polyfit returns coefficients of the highest power first
        element = poly_coefficients[i]
        w5.append(element[0])
        w4.append(element[1])
        w3.append(element[2])
        w2.append(element[3])
        w1.append(element[4])



    return w1,w2,w3,w4,w5


def Calling_csv(sheets):
        
        series_data_clean(sheets) 
        f1,f2 = main_method(sheets)
        f3,f4 = get_velocities(sheets)
        f5=feature_DWT(sheets)
    
        f6,f7,f8,f9,f10=polyfit_feature(sheets)
   
        f1=np.transpose(f1)
        f2=np.transpose(f2)
        f6=np.transpose(f6)
        f7=np.transpose(f7)
        f8=np.transpose(f8)
        f9=np.transpose(f9)
        f10=np.transpose(f10)
    
  
    
        final_list = [f1,f2,f3,f4,f6,f7,f8,f9,f10]
        final_list=np.transpose(final_list)
        final_list_2 = pd.DataFrame(final_list)
        df_out1=pd.concat([final_list_2,f5],axis=1)
        
        return df_out1

def run_code_here(filepath):
    s_data=pd.read_csv(filepath)
    temp=s_data.interpolate(method='quadratic')
    temp_s=temp.interpolate(method='linear')
    
    out1=temp_s.dropna()
    out1 = Calling_csv(filepath)

    out=out1.iloc[1:,:]
    #print("OUT",out.shape)
#     out_y = out1.iloc[1:,-1]
    
    standardized = StandardScaler().fit_transform(out)
    #print(standardized.shape)
    pkl_file = pickle.load(open('pca.pkl', 'rb'))
    
    new_pca=pkl_file.transform(standardized)
    
    
    with open("gradient_booster.pkl", 'rb') as file:
        GBC_unpickle=pickle.load(file)
    
    with open("Random_forest.pkl", 'rb') as file:
        RF_unpickle=pickle.load(file)
    with open("Decision tree.pkl", 'rb') as file:
        DT_unpickle=pickle.load(file)
    
    with open("KNN.pkl", 'rb') as file:
        KNN_unpickle=pickle.load(file)
        
    X =new_pca
    y_predicted_GBC = GBC_unpickle.predict(X)
    print("Gradient Boosting Classifier:",y_predicted_GBC)
    y_predicted_RF=RF_unpickle.predict(X)
    print("Random Forest Classiefier:",y_predicted_RF)
    y_predicted_DT=DT_unpickle.predict(X)
    print("Decision Tree Classifier:",y_predicted_DT)
    y_predicted_KNN = KNN_unpickle.predict(X)
    print("K Nearest Neighbours:",y_predicted_KNN)
    
#     print("Classification Report for different Classifiers")
#     print("Gradient Boosting Classifier")
#     print(classification_report(out_y, y_predicted_GBC.round()))
#     print("Random Forest")
#     print(classification_report(out_y, y_predicted_RF))
#     print("Decision Tree")
#     print(classification_report(out_y, y_predicted_DT))
#     print("K-Nearest Neighbour")
#     print(classification_report(out_y, y_predicted_KNN.round()))
    
#     gbc_acc=[]
#     rf_acc=[]
#     td_acc=[]
#     knn_acc=[]
#     print(" Average Accuracy for all the classifiers")
#     print(gbc_acc.append(y_predicted_GBC.round()))
#     print(rf_acc.append( y_predicted_RF))
#     td_acc.append(y_predicted_DT)
#     knn_acc.append(y_predicted_KNN.round())
#     print(np.mean([gbc_acc, rf_acc, td_acc, knn_acc]))
#     gbc_acc.append(accuracy_score(out_y, y_predicted_GBC.round()))
#     rf_acc.append(accuracy_score(out_y, y_predicted_RF))
#     td_acc.append(accuracy_score(out_y, y_predicted_DT))
#     knn_acc.append(accuracy_score(out_y, y_predicted_KNN.round()))
#     print(np.mean([gbc_acc, rf_acc, td_acc, knn_acc]))
    
    
if __name__ == "__main__":
    filepath = input("Enter your file path here")
    run_code_here(filepath)
    
    
    


# In[ ]:





# In[ ]:




