#RandomForest model and tuning
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\Hitters.csv")
datas=datas.dropna()
dms=pd.get_dummies(datas[["League","NewLeague","Division"]])
y=datas["Salary"]
x_=datas.drop(["Salary","League","NewLeague","Division"],axis=1).astype("float64")
x=pd.concat([x_,dms],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=99)

RFR=RandomForestRegressor()
RFR_params={
    "n_estimators":[50,100,150,200],
    "max_depth":[10,20,30,40],
    "min_samples_split":[2,4,6,8],
    "max_features":[1,2,3,4]
}
RFR_cv=GridSearchCV(RFR,RFR_params,cv=10,n_jobs=-1,verbose=2)
RFR_cv.fit(x_train,y_train)
n_estimators=RFR_cv.best_params_["n_estimators"]
max_depth=RFR_cv.best_params_["max_depth"]
min_samples_split=RFR_cv.best_params_["min_samples_split"]
max_features=RFR_cv.best_params_["max_features"]
RFR_tuned=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,max_features=max_features)
RFR_tuned.fit(x_train,y_train)
predict=RFR_tuned.predict(x_test)
RMSE=np.sqrt(mean_squared_error(y_test,predict))
print(RMSE)


#veriler içerisindeki en önemli verileri döndüren komutlar
Importance=pd.DataFrame({"Importance":RFR_tuned.feature_importances_*100},
                        index=x_train.columns)
Importance.sort_values(by="Importance",
                       axis=0,
                       ascending=True).plot(kind="barh",color="r")
plt.xlabel("Variable Importance")
plt.gca().legend_=None
plt.show()












