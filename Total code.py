#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[38]:
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
import csv
import tsfresh
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import sklearn.svm
from sklearn.model_selection import KFold
# from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import tree
from sklearn.utils import shuffle
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras import layers
import sklearn
import pickle

pca=sklearn.decomposition.PCA(n_components=5)


#Feature 1
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


def No_Meal_tryout(df_out_final_meal,df_out_final_no_meal):
    
    standardized = StandardScaler().fit_transform(df_out_final_meal)
    standardized_no_meal = StandardScaler().fit_transform(df_out_final_no_meal)
    pca=PCA(n_components=5)
    pc=pca.fit_transform(standardized)
    pickle.dump(pca, open('pca.pkl','wb'))
    pca_no_meal=pca.transform(standardized_no_meal)
    
    principalDf = pd.DataFrame(data = pc , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])

    principalDf.insert( 5,"Label", 1)
    #print(principalDf.shape)
    
    principalDf_no_meal = pd.DataFrame(data = pca_no_meal , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])
    principalDf_no_meal.insert( 5,"Label", 0)
    #print(principalDf_no_meal.shape)
    merged_df_no_meal=pd.concat([principalDf, principalDf_no_meal])
    
    merged_df_no_meal.to_csv("merged_df.csv")
    
    
    
    #return merged_df_no_meal,pc
    return merged_df_no_meal

def run_code_here(filepath):
    s_data=pd.read_csv(filepath)
    temp=s_data.interpolate(method='quadratic')
    temp_s=temp.interpolate(method='linear')
    
    out1=temp_s.dropna()

    out=out1.iloc[1:,:5]
    
    out_y = out1.iloc[1:,-1]
    
    standardized = StandardScaler().fit_transform(out)
    pkl_file = pickle.load(open('pca.pkl', 'rb'))
    
    new_pca=pca.fit_transform(standardized)
    
    
    with open("gradient_booster.pkl", 'rb') as file:
        GBC_unpickle=pickle.load(file)
    
    with open("Random_forest.pkl", 'rb') as file:
        RF_unpickle=pickle.load(file)
    with open("Decision tree.pkl", 'rb') as file:
        DT_unpickle=pickle.load(file)
    
    with open("KNN.pkl", 'rb') as file:
        KNN_unpickle=pickle.load(file)
        
    X =out
    y_predicted_GBC = GBC_unpickle.predict(X)
    y_predicted_RF=RF_unpickle.predict(X)
    y_predicted_DT=DT_unpickle.predict(X)
    y_predicted_KNN = KNN_unpickle.predict(X)
    
    print("Classification Report for different Classifiers")
    print("Gradient Boosting Classifier")
    print(classification_report(out_y, y_predicted_GBC.round()))
    print("Random Forest")
    print(classification_report(out_y, y_predicted_RF))
    print("Decision Tree")
    print(classification_report(out_y, y_predicted_DT))
    print("K-Nearest Neighbour")
    print(classification_report(out_y, y_predicted_KNN.round()))
    
    
    print(" Average Accuracy for all the classifiers")
    gbc_acc=accuracy_score(out_y, y_predicted_GBC.round())
    rf_acc=accuracy_score(out_y, y_predicted_RF)
    td_acc=accuracy_score(out_y, y_predicted_DT)
    knn_acc=accuracy_score(out_y, y_predicted_KNN.round())
    print(np.mean([gbc_acc, rf_acc, td_acc, knn_acc]))
    
    
    
    
    
    
    
#Graident Boosting Classifier
def GBC(X,y):
    


    scores=[]
    

    kfold = KFold(4,True,1)
    for train, test in kfold.split(X):
        
        X_train=X.iloc[train]
        X_test=X.iloc[test]
        y_train,y_test=y.iloc[train],y.iloc[test]
         
        clf = GradientBoostingClassifier(learning_rate=0.1,n_estimators=32,max_depth=4)
        
        clf.fit(X_train, y_train)
        
        #predicted=clf.predict(X_test)
        
        #scores.append(accuracy_score(y_test, predicted.round(), normalize=False))
    random_pkl = "gradient_booster.pkl"
    with open(random_pkl, 'wb') as file:
        pickle.dump(clf, file)  
#     print(" Mean scores for the Gradient Boosting Classifer of : ", np.mean(scores))
#     print("CLassification Report for Gradient Boosting Classifier : ")
#     print(classification_report(y_test,predicted))

def KNN(X,y):
    


    scores=[]
    

    kfold = KFold(4,True,1)
    for train, test in kfold.split(X):
        
        X_train=X.iloc[train]
        X_test=X.iloc[test]
        y_train,y_test=y.iloc[train],y.iloc[test]
         
        clf = clf=KNeighborsClassifier(n_neighbors=2,p=1)
        
        clf.fit(X_train, y_train)
        
        #predicted=clf.predict(X_test)
        
        #scores.append(accuracy_score(y_test, predicted.round(), normalize=False))
    random_pkl = "KNN.pkl"
    with open(random_pkl, 'wb') as file:
        pickle.dump(clf, file)  


#Random Forest
def Random_forest(X,y):
        
    
    score=[]
    kfold = KFold(4,True,1)
    for train, test in kfold.split(X):
        
        #print(type(train))
        X_train=X.iloc[train]
        X_test=X.iloc[test]
        y_train,y_test=y.iloc[train],y.iloc[test]
        clf = RandomForestClassifier(n_estimators='warn', max_depth=8,random_state=0)
        clf.fit(X_train, y_train)
    classi_2 = "Random_forest.pkl"
    with open(classi_2, 'wb') as file:
        pickle.dump(clf, file)  
       
#         y_pred = clf.predict(X_test)
#         score.append(accuracy_score(y_test, y_pred, normalize=False))
        
#     print("Mean scores for the Random Forest Classifier : ",(np.mean(score)))
    #print(" Classification Report for Random Forest Classifier : ")
#     print(classification_report(y_test,y_pred))
      

    
    

#Decision Tree
def DecisionTree(X,y):
#     print("DecisionTree Classifier ")
    kfold = KFold(11,True,1)
    for train, test in kfold.split(X):
        X_train=X.iloc[train]
        X_test=X.iloc[test]
        y_train,y_test=y.iloc[train],y.iloc[test]

        clf = DecisionTreeClassifier(max_depth=3, min_samples_split=0.4, min_samples_leaf=0.1,max_features=5)
    
            #train the algorithm on training data and predict using the testing data
        clf.fit(X_train, y_train)
    classi_3 = "Decision tree.pkl"
    with open(classi_3, 'wb') as file:
        pickle.dump(clf, file)  
#     pred=clf.predict(X_test)
#             #print the accuracy score of the model
#     print("Decision Tree Accuracy : ",accuracy_score(y_test, pred, normalize = True) * 100)
#     print(" Classification Report for Decision Tree Classifier : ")
#     print(classification_report(y_test, pred))



    
#Artificial Neural Networks

def ANN(df):
    

    X=df.iloc[:,1:-1]
    y=df["Label"]
    print("X",X)
    print("y",y.shape)
    kfold = KFold(4,True,1)
    acc=[]
    test_acc=[]
    scores=[]
    f1=[]
    for train, test in kfold.split(X):
        
        X_train=X.iloc[train]
        X_test=X.iloc[test]
        y_train,y_test=y.iloc[train],y.iloc[test]
        print(X_train.shape, y_train.shape)

        model = Sequential()
        model.add(Dense(16, input_dim=5))
      
        model.add(Activation("relu"))
        model.add(Dense(16))
       
        model.add(Activation("relu"))

        model.add(Dense(1))
       
        model.add(Activation("sigmoid"))


        model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics=['accuracy'])
        
        print( y_train)
        model.fit(X_train, y_train, epochs=100, batch_size=4)
    classi_4 = "ann.pkl"
    
      
    _, accuracy = model.evaluate(X, y)
    print("***********************",accuracy)
    with open(classi_4, 'wb') as file:
        pickle.dump(model, file)
    
      
#         predicted=model.predict(X_test)
       
#         print(model.evaluate(X_test, y_test, verbose=0))
#         _,accuracy = model.evaluate(X_test, y_test, verbose=0)
#         test_acc.append(accuracy)
        

#         print(sklearn.metrics.classification_report(y_test, predicted.round()))
#     print(acc)
#     print("Train Accuracy", np.mean(test_acc))
#     # print("F1", np.mean(f1))
#     # print("Scores", np.mean(scores))
#     test_loss, test = model.evaluate(X_test, y_test)
#     print("Test Accuracy", np.mean(test_acc))

def Calling_Classifier():
    
    csv_file="merged_df.csv"
    df=pd.read_csv(csv_file)
    
    X=df.iloc[1:,:5]
    y=df["Label"]
    y=y[1:]
    Random_forest(X,y)
    
    GBC(X,y)   
    
    DecisionTree(X,y)    
    
    #ANN(df)
    
    KNN(X,y)
    


if __name__ == "__main__":

    
    meal_data_1=Calling_csv('mealData1.csv')
    meal_data_2=Calling_csv('mealData2.csv')
    meal_data_3=Calling_csv('mealData3.csv')
    meal_data_4=Calling_csv('mealData4.csv')
    meal_data_5=Calling_csv('mealData5.csv')
    
    no_meal_data_1=Calling_csv('Nomeal1.csv')
    no_meal_data_2=Calling_csv('Nomeal2.csv')
    no_meal_data_3=Calling_csv('Nomeal3.csv')
    no_meal_data_4=Calling_csv('Nomeal4.csv')
    no_meal_data_5=Calling_csv('Nomeal5.csv')
    
    df_out_final_meal=pd.DataFrame()
    df_out_final_meal= pd.concat([meal_data_1,meal_data_2,meal_data_3,meal_data_4,meal_data_5])
    
    
    df_out_final_no_meal=pd.DataFrame()
    df_out_final_no_meal= pd.concat([no_meal_data_1,no_meal_data_2,no_meal_data_3,no_meal_data_4,no_meal_data_5])
    
    
    merged_df_no_meal=No_Meal_tryout(df_out_final_meal,df_out_final_no_meal)
    Calling_Classifier()
    run_code_here('merged_df.csv')

    
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:




