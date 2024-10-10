
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


from joblib import load, dump
from sklearn.model_selection import cross_val_score, train_test_split,  GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



def getClasses(directory):
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            image_paths.append(os.path.join(root, file))

    classes = [os.path.basename(os.path.dirname(image)) for image in image_paths]

    classes = np.repeat(classes, 196, axis=0)

    return classes





image_paths = "./new_training/"
y=sorted(getClasses(image_paths))
X=pd.read_csv("./training_embd_oi16/embeddings_s1_new_extd.csv")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
rf = RandomForestClassifier(random_state=42, n_jobs=-1)







#Search of best estimators
"""
param_grid = {'n_estimators': [50, 100, 200], 

    'min_samples_split': [2, 5, 10],  


    'bootstrap': [True, False], 

    'criterion' :['gini', 'entropy'], 

     #'max_depth': [10, 20, 30, 40, 50, 60, 100, None] 

} 

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

#
#RandomForestClassifier(bootstrap=False, min_samples_split=5, n_estimators=200,
 #                      n_jobs=-1, random_state=42)

"""


rf_best=RandomForestClassifier(bootstrap=False, min_samples_split=5, n_estimators=200,
                    n_jobs=-1, random_state=42)
rf_best.fit(X_train, y_train)

#Validation phase
y_pred_val = rf_best.predict(X_val)
accuracy=accuracy_score(y_val,y_pred_val)
print(f'Accuracy:{accuracy}')


#Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_val, y_pred_val)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(class_labels))
disp.plot()
plt.show()





"""
#Save the model
dump(rf_best, 'best_rf_supervised_aoi.joblib')

rf_best = load('best_rf_supervised_aoi.joblib')
"""






#Embeddings of patches (16px16p) of our AOI sorted in the following order: 
#geographical position of the sub-image from which the patch is extracted, latitude(north to south), longitude(west to east) 

X_test=np.load("aoi_embd_oi16.npy")
y_pred = rf_best.predict(X_test)
y_pred_proba = rf_best.predict_proba(X_test)
results_df = pd.DataFrame({
    'Prediction': y_pred,
    'Probability(0)': y_pred_proba[:, 0], 
    'Probability(1)': y_pred_proba[:, 1],
    'Probability(2)': y_pred_proba[:, 2] 
    
})
print(results_df)






#Find the right threshold of prediction scores for classes (“Forest”, ‘Lawn’, ‘Mineral class’.)

y_pred_best = []
for proba in y_pred_proba:
    
    if proba[1] >=0.6:
        y_pred_best.append("Forest")
    else:
        if proba[0]>=0.5:
            y_pred_best.append("Mineral class")
        else :
            if proba[2]>=0.6:
                y_pred_best.append("Lawn")
            else :
                y_pred_best.append("Unknown")






import geopandas as gpd
shp=gpd.read_file("clustering_aoi_newsupervised.shp")
shp['f6-m5-l6']=y_pred_best
shp['cluster']=y_pred
shp.to_file("clustering_aoi_adjusted.shp")
