import pandas as pd 
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

df = pd.read_csv('data.csv')
X = df[["glucose","bloodpressure"]]
Y= df["diabetes"]

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

gn = GaussianNB()
gn.fit(X_train,Y_train)

y_pred = gn.predict(X_test)

print("Accuracy is ---->" , accuracy_score(Y_test, y_pred))

X=df[["glucose","bloodpressure"]]
Y=df["diabetes"]

x_train_1 , x_test_1 , y_train_1 , y_test_1 = train_test_split(X , Y, test_size=0.25 , random_state=42)
sc = StandardScaler()

x_train_1 = sc.fit_transform(x_train_1)
x_test_1 = sc.fit_transform(x_test_1)

model_1 = LogisticRegression(random_state=0)
model_1.fit(x_train_1,y_train_1)

y_pred_1 = model_1.predict(x_test_1)

accuracy = accuracy_score(y_test_1, y_pred)
print("accuracy using logistic==+=====+++====",accuracy)