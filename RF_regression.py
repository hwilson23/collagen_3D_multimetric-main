import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor



#df = pd.read_csv("final_dataframe.csv")
df = pd.read_csv("final_dataframe_byslice.csv")
print(df['type_asmmean.'])
df['gel_type'] = df['type_asmmean.'].map({'shg': 0, 'flu': 1})
df['gel_concen'] = df['concentration_asmmean.'].map({'1mgml':1, '2mgml':2, '3mgml':3})
#remove unnecessary columns
df = df.drop(['TotalImageArea','Unnamed: 13','Unnamed: 14'],axis=1)
#reduce slices and sampling
df = df[df['slice']%5==0]
print(len(df))

print(df[df.isna().any(axis=1)])
#print(df.columns.values)
df = df.select_dtypes(include=['number'])
#print(df.columns.values)

X = df.drop(columns=['gel_concen'])
y = df['gel_concen']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

single_data = X_test.iloc[0].values.reshape(1, -1)
predicted_value = rf_regressor.predict(single_data)
print(f"Predicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")




importance = pd.Series(rf_regressor.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)
importance.plot(kind='barh')
plt.show()
print("Done :)")

