import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

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
    # Предсказания на обучающей и тестовой выборках
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Вычисление метрик с использованием функций из sklearn
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    # Вывод результатов
    print(f"\n{model_name}:")
    print("\nМетрики sklearn:")
    print(f"R² (обучающая выборка): {train_r2:.4f}")
    print(f"R² (тестовая выборка): {test_r2:.4f}")
    print(f"MAE (обучающая выборка): {train_mae:.4f}")
    print(f"MAE (тестовая выборка): {test_mae:.4f}")
    print(f"MSE (обучающая выборка): {train_mse:.4f}")
    print(f"MSE (тестовая выборка): {test_mse:.4f}")
    print(f"RMSE (обучающая выборка): {train_rmse:.4f}")
    print(f"RMSE (тестовая выборка): {test_rmse:.4f}")

    # Визуализация результатов
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'{model_name} - Обучающая выборка\nR² = {train_r2:.4f}, RMSE = {train_rmse:.4f}')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'{model_name} - Тестовая выборка\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}')
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name.replace(" ", "_").lower()}_scatter.png')
    plt.close()
    
    # Возвращаем словарь с метриками для сравнения моделей
    return {
        'model_name': model_name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
    }

# Создание и обучение модели GradientBoostingRegressor
gb_reg = GradientBoostingRegressor(random_state=42)
gb_reg.fit(X_train, y_train)

# Оценка качества модели
gb_results = evaluate_model(gb_reg, X_train, X_test, y_train, y_test, 'GradientBoostingRegressor')

# Сохранение результатов в CSV
gb_results_df = pd.DataFrame({
    'Model': ['GradientBoostingRegressor'],
    'Train R2': [gb_results['train_r2']],
    'Test R2': [gb_results['test_r2']],
    'Train RMSE': [gb_results['train_rmse']],
    'Test RMSE': [gb_results['test_rmse']]
})
gb_results_df.to_csv('results/gb_results.csv', index=False)
print('Результаты GradientBoostingRegressor сохранены в results/gb_results.csv')

# Сохранение модели с помощью pickle
with open('models/model_ml2.pkl', 'wb') as f:
    pickle.dump(gb_reg, f)
print('Модель GradientBoostingRegressor сохранена в models/model_ml2.pkl')