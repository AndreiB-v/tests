# В данном уроке мы построим модель,
# позволяющую прогнозировать данные на следующую неделю, основываясь на прошлые

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

df_1['date'] = df_1['date'].apply(lambda x: pd.to_datetime(x, format='%d-%m-%Y'))
df_2['date'] = df_2['date'].apply(lambda x: pd.to_datetime(x, format='%d-%m-%Y'))

# Агрегация по неделям
weekly1 = df_1.resample('W', on='date')['summ'].sum()
weekly2 = df_2.resample('W', on='date')['summ'].sum()

# Создание общего DataFrame
df = pd.DataFrame({
    'income': weekly1,
    'outcome': weekly2
}).fillna(0)

# Целевая переменная - чистый поток
df['net_flow'] = df['income'] - df['outcome']

# Создание признаков (лаги, скользящие средние и т.д.)
for i in [1, 2, 3, 4]:
    df[f'lag_{i}'] = df['net_flow'].shift(i)

df['rolling_mean_4'] = df['net_flow'].rolling(4).mean().shift(1)
df = df.dropna()

# Разделение на признаки и целевую переменную
X = df.drop('net_flow', axis=1)
y = df['net_flow']

# Разделение на train/test с учетом временных рядов
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Инициализация моделей
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Обучение и оценка
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {'MAE': mae, 'RMSE': rmse}

    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}")


# Параметры для оптимизации

# n_estimators — количество деревьев в ансамбле (чем больше, тем точнее, но дольше обучение).
# max_depth — максимальная глубина каждого дерева (глубже = сложнее модель, риск переобучения).
# learning_rate — шаг градиентного бустинга (меньше = медленнее, но точнее обучение).

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# TimeSeriesSplit для кросс-валидации

# Так как данные — временной ряд, обычная случайная разбивка (KFold) не подходит.
# TimeSeriesSplit делит данные последовательно, сохраняя временной порядок

tscv = TimeSeriesSplit(n_splits=5)

# Выбор лучшие параметры для модели по сетке

# Данный участок кода находит лучшие параметры, подставляя разные гипер-параметры
# так как он ищет максимальное значение, устанавливаем neg_mean_absolute_error
# она обращает mean_absolute_error в отрицательное значение

grid_search = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)

# Модель с лучшими параметрами, ее можно будет сохранить
best_model = grid_search.best_estimator_

# Строим кривую обучения

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X_train,
    y_train,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, -train_scores.mean(axis=1), label='Training error')
plt.plot(train_sizes, -val_scores.mean(axis=1), label='Validation error')
plt.xlabel('Training set size')
plt.ylabel('MAE')
plt.title('Learning curves')
plt.legend()
plt.show()


# В итоге данного урока была построена, обучена и проверенна модель,
# позволяющая предсказывать значения на следующую неделю

# Пример входных данных для прогноза:
input_data = pd.DataFrame({
    'income': [50000],       # Доходы за текущую неделю
    'outcome': [30000],      # Расходы за текущую неделю
    'lag_1': [20000],        # net_flow за предыдущую неделю
    'lag_2': [15000],        # net_flow за неделю -2
    'lag_3': [10000],        # net_flow за неделю -3
    'lag_4': [5000],         # net_flow за неделю -4
    'rolling_mean_4': [12500] # Среднее за последние 4 недели
})

# net_flow = доходы - расходы (прибыль)

prediction = model.predict(input_data)  # Прогноз net_flow на следующую неделю

# Для переобучения подобной модели на новых данных можно использовать скользящее окно

window_size = 52  # Например, последний год (52 недели)
X_recent = X_new.iloc[-window_size:]
y_recent = y_new.iloc[-window_size:]

model.fit(X_recent, y_recent)
