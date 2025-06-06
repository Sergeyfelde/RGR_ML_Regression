import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Для Optuna и Hyperopt

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Создание директорий для сохранения результатов и моделей, если их нет
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Загрузка предобработанного датасета
data = pd.read_csv('data/EDA_regression.csv')

# Выделение целевого признака и предикторов
y = data['price']
X = data.drop('price', axis=1)

# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Оценка качества модели с использованием различных метрик"""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    print(f"\n{model_name}:")
    print(f"R2 train: {train_r2:.4f}, R2 test: {test_r2:.4f}")
    print(f"RMSE train: {train_rmse:.4f}, RMSE test: {test_rmse:.4f}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'{model_name} - Обучающая выборка')
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'{model_name} - Тестовая выборка')
    plt.tight_layout()
    plt.savefig(f'results/{model_name.replace(" ", "_").lower()}_scatter.png')
    plt.close()
    return {'model_name': model_name, 'train_r2': train_r2, 'test_r2': test_r2, 'train_rmse': train_rmse, 'test_rmse': test_rmse}

# 1. RandomizedSearchCV
mlp = MLPRegressor(max_iter=1000, random_state=42, early_stopping=True)
param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (50,50)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['adaptive'],
    'early_stopping': [True],
}
rand_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=20, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
print(f"Лучшие параметры (RandomizedSearchCV): {rand_search.best_params_}")
mlp_rand_results = evaluate_model(rand_search.best_estimator_, X_train, X_test, y_train, y_test, 'MLPRegressor (RandomizedSearchCV)')
with open('models/model_ml6_rand.pkl', 'wb') as f:
    pickle.dump(rand_search.best_estimator_, f)


# Сохранение результатов в CSV
results_df = pd.DataFrame([
    {'Model': 'MLPRegressor (RandomizedSearchCV)', **mlp_rand_results}
])
results_df.to_csv('results/mlp_results.csv', index=False)
print('Результаты MLPRegressor сохранены в results/mlp_results.csv')
print('Модель MLPRegressor сохранена в models/model_ml6_rand.pkl')
