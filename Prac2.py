import pandas as pd 
import matplotlib.pyplot as plt from sklearn.model_selection import train_test_split from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix from sklearn.preprocessing import LabelEncoder 
 
df = pd.read_csv('weather.csv') 
 
 
print("Linear Regression: Predicting MaxTemp") 
 
 
X_lin = df[['MinTemp']].values  
y_lin = df['MaxTemp'].values 
 
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42) 
 
lr_model = LinearRegression() 
lr_model.fit(X_train_l, y_train_l) 
 
y_pred_l = lr_model.predict(X_test_l) print(f"Mean Squared Error: {mean_squared_error(y_test_l, y_pred_l):.2f}\n") 
 
plt.scatter(X_test_l, y_test_l, color='blue', label='Actual') plt.plot(X_test_l, y_pred_l, color='red', linewidth=2, label='Regression Line') plt.xlabel("Min Temperature") plt.ylabel("Max Temperature") plt.title("MinTemp vs MaxTemp") plt.legend() 
plt.show() 
 
print("Logistic Regression: Predicting Rain Tomorrow") 
 
 
le = LabelEncoder() df['RainToday'] = le.fit_transform(df['RainToday'].astype(str)) df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'].astype(str)) 
 
features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 
'Humidity3pm'] 
X_log = df[features].fillna(0)  
y_log = df['RainTomorrow'] 
 
X_train_log, X_test_log, y_train_log, y_test_log = 
train_test_split(X_log, y_log, test_size=0.2, random_state=42) 
 
log_model = LogisticRegression(max_iter=1000) log_model.fit(X_train_log, y_train_log) 
 
y_pred_log = log_model.predict(X_test_log) 
 
print(f"Accuracy: {accuracy_score(y_test_log, y_pred_log):.2f}") print("Confusion Matrix:") print(confusion_matrix(y_test_log, y_pred_log)) 
