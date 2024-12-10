from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split, cross_val_score

# Разделение данных
X = ...
y = ...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = [RandomForestRegressor(),
          LinearRegression(),
          Ridge(),
          Lasso(max_iter=1000000),
          ElasticNet(alpha=0.001, l1_ratio=0.0001, max_iter=999999999)]

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    score = r2_score(y_test, y_pred)
    print(score, mse)

# Получаем важность признаков
feature_importance = pd.DataFrame({
    'Признак': X.columns,
    'Важность': model.feature_importances_
}).sort_values('Важность', ascending=False)

# Визуализация
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Признак'][:15], feature_importance['Важность'][:15])
plt.title('Топ-15 важных признаков')
plt.show()
