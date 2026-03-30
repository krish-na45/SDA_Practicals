import pandas as pd from sklearn.model_selection import train_test_split from sklearn.tree import DecisionTreeClassifier from sklearn.ensemble import RandomForestClassifier from sklearn.metrics import accuracy_score from sklearn.preprocessing import LabelEncoder 
 
 
df = pd.read_csv('weather.csv') 
 
 
le = LabelEncoder() df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'].astype(str)) 
 
 
features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 
'Humidity3pm'] 
X = df[features].fillna(0)  
y = df['RainTomorrow'] 

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
 
dt = DecisionTreeClassifier(random_state=42) dt.fit(X_train, y_train) 
 
 
dt_train_acc = accuracy_score(y_train, dt.predict(X_train)) dt_test_acc = accuracy_score(y_test, dt.predict(X_test)) 
 
print(f"Decision Tree - Training Accuracy: {dt_train_acc:.2f}") print(f"Decision Tree - Testing Accuracy: {dt_test_acc:.2f}") 
 
 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train) 
 
rf_train_acc = accuracy_score(y_train, rf.predict(X_train)) rf_test_acc = accuracy_score(y_test, rf.predict(X_test)) 
 
print(f"\nRandom Forest - Training Accuracy: {rf_train_acc:.2f}") print(f"Random Forest - Testing Accuracy: {rf_test_acc:.2f}") 
 
 
print("\nModel Comparison") print(f"Accuracy Gap (DT): {dt_train_acc - dt_test_acc:.4f}") print(f"Accuracy Gap (RF): {rf_train_acc - rf_test_acc:.4f}") 
 
