import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Создание директорий для сохранения визуализаций, если их нет
os.makedirs('visualizations', exist_ok=True)

# Функция для загрузки результатов всех моделей
def load_all_results():
    """Загрузка результатов всех моделей из CSV-файлов"""
    results_files = [
        'results/elastic_results.csv',
        'results/gb_results.csv',
        'results/catboost_results.csv',
        'results/bagging_results.csv',
        'results/stacking_results.csv'
    ]
    
    all_results = []
    for file in results_files:
        if os.path.exists(file):
            results = pd.read_csv(file)
            all_results.append(results)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

# Функция для создания сравнительных визуализаций
def create_comparison_visualizations():
    """Создание сравнительных визуализаций для всех моделей"""
    results_df = load_all_results()
    
    if results_df.empty:
        print("Файлы с результатами не найдены.")
        return
    
    # 1. Сравнение R² на тренировочной и тестовой выборках
    plt.figure(figsize=(12, 6))
    
    bar_width = 0.35
    index = np.arange(len(results_df['Model']))
    
    plt.bar(index, results_df['Train R2'], bar_width, label='Обучающая выборка')
    plt.bar(index + bar_width, results_df['Test R2'], bar_width, label='Тестовая выборка')
    
    plt.xlabel('Модель')
    plt.ylabel('R²')
    plt.title('Сравнение R² разных моделей')
    plt.xticks(index + bar_width / 2, results_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/r2_comparison.png')
    plt.close()
    
    # 2. Сравнение RMSE на тренировочной и тестовой выборках
    plt.figure(figsize=(12, 6))
    
    plt.bar(index, results_df['Train RMSE'], bar_width, label='Обучающая выборка')
    plt.bar(index + bar_width, results_df['Test RMSE'], bar_width, label='Тестовая выборка')
    
    plt.xlabel('Модель')
    plt.ylabel('RMSE')
    plt.title('Сравнение RMSE разных моделей')
    plt.xticks(index + bar_width / 2, results_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/rmse_comparison.png')
    plt.close()
    
    # 3. Сравнение показателей качества моделей (тепловая карта)
    metrics_df = results_df.set_index('Model')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.4f')
    plt.title('Сравнение метрик всех моделей')
    plt.tight_layout()
    plt.savefig('visualizations/metrics_heatmap.png')
    plt.close()
    
    print("Сравнительные визуализации сохранены в директорию 'visualizations'.")

# Функция для создания визуализаций распределения ошибок
def create_error_distributions():
    """Создание визуализаций распределения ошибок для всех моделей"""
    # Этот код предполагает, что у нас есть доступ к предсказаниям моделей
    # Для демонстрации создадим синтетические данные
    
    # Загрузка данных
    try:
        data = pd.read_csv('data/EDA_regression.csv')
        
        # Выделение целевого признака и предикторов
        y = data['price']
        X = data.drop('price', axis=1)
        
        # Разбиение данных на обучающую и тестовую выборки
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Загрузка всех моделей
        import pickle
        from catboost import CatBoostRegressor
        
        models = {}
        
        # Загрузка модели ML1 (ElasticNet)
        try:
            models['ML1: ElasticNet'] = pickle.load(open('models/model_ml1.pkl', 'rb'))
        except:
            print("Модель ML1 не найдена")
        
        # Загрузка модели ML2 (GradientBoosting)
        try:
            models['ML2: GradientBoosting'] = pickle.load(open('models/model_ml2.pkl', 'rb'))
        except:
            print("Модель ML2 не найдена")
        
        # Загрузка модели ML3 (CatBoost)
        try:
            cb_model = CatBoostRegressor()
            cb_model.load_model('models/model_ml3.cbm')
            models['ML3: CatBoost'] = cb_model
        except:
            print("Модель ML3 не найдена")
        
        # Загрузка модели ML4 (Bagging)
        try:
            models['ML4: Bagging'] = pickle.load(open('models/model_ml4.pkl', 'rb'))
        except:
            print("Модель ML4 не найдена")
        
        # Загрузка модели ML5 (Stacking)
        try:
            models['ML5: Stacking'] = pickle.load(open('models/model_ml5.pkl', 'rb'))
        except:
            print("Модель ML5 не найдена")
        
        # Если есть хотя бы одна модель
        if models:
            # Создание гистограмм ошибок для каждой модели
            plt.figure(figsize=(15, 10))
            
            for i, (name, model) in enumerate(models.items(), 1):
                y_pred = model.predict(X_test)
                errors = y_test - y_pred
                
                plt.subplot(2, 3, i)
                sns.histplot(errors, kde=True)
                plt.title(f'Распределение ошибок - {name}')
                plt.xlabel('Ошибка')
                plt.ylabel('Частота')
            
            plt.tight_layout()
            plt.savefig('visualizations/error_distributions.png')
            plt.close()
            
            # Создание QQ-плотов для проверки нормальности распределения ошибок
            import scipy.stats as stats
            
            plt.figure(figsize=(15, 10))
            
            for i, (name, model) in enumerate(models.items(), 1):
                y_pred = model.predict(X_test)
                errors = y_test - y_pred
                
                plt.subplot(2, 3, i)
                stats.probplot(errors, dist="norm", plot=plt)
                plt.title(f'QQ-график ошибок - {name}')
            
            plt.tight_layout()
            plt.savefig('visualizations/error_qq_plots.png')
            plt.close()
            
            print("Визуализации распределения ошибок сохранены в директорию 'visualizations'.")
        else:
            print("Ни одной модели не найдено.")
    
    except Exception as e:
        print(f"Ошибка при создании визуализации ошибок: {e}")

# Функция для создания визуализации важности признаков
def create_feature_importance():
    """Создание визуализации важности признаков для моделей, которые поддерживают это"""
    try:
        # Загрузка данных
        data = pd.read_csv('data/EDA_regression.csv')
        
        # Выделение целевого признака и предикторов
        X = data.drop('price', axis=1)
        
        # Получение имен признаков
        feature_names = X.columns.tolist()
        
        # Загрузка моделей, которые поддерживают извлечение важности признаков
        import pickle
        from catboost import CatBoostRegressor
        
        feature_importance_models = {}
        
        # Загрузка модели ML2 (GradientBoosting)
        try:
            model = pickle.load(open('models/model_ml2.pkl', 'rb'))
            feature_importance_models['ML2: GradientBoosting'] = model
        except:
            pass
        
        # Загрузка модели ML3 (CatBoost)
        try:
            model = CatBoostRegressor()
            model.load_model('models/model_ml3.cbm')
            feature_importance_models['ML3: CatBoost'] = model
        except:
            pass
        
        # Загрузка модели ML4 (Bagging с базовым алгоритмом DecisionTree)
        try:
            model = pickle.load(open('models/model_ml4.pkl', 'rb'))
            if hasattr(model, 'feature_importances_') or hasattr(model.estimators_[0], 'feature_importances_'):
                feature_importance_models['ML4: Bagging'] = model
        except:
            pass
        
        # Если есть хотя бы одна модель с важностью признаков
        if feature_importance_models:
            plt.figure(figsize=(12, len(feature_importance_models) * 4))
            
            for i, (name, model) in enumerate(feature_importance_models.items(), 1):
                plt.subplot(len(feature_importance_models), 1, i)
                
                # Извлечение важности признаков
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif name == 'ML4: Bagging' and hasattr(model.estimators_[0], 'feature_importances_'):
                    # Для бэггинга усредним важность признаков по всем базовым моделям
                    importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
                else:
                    continue
                
                # Сортировка по убыванию важности
                indices = np.argsort(importances)[::-1]
                
                # Построение графика
                plt.barh(range(len(importances)), importances[indices])
                plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
                plt.title(f'Важность признаков - {name}')
                plt.xlabel('Важность')
                plt.tight_layout()
            
            plt.savefig('visualizations/feature_importance.png')
            plt.close()
            
            print("Визуализация важности признаков сохранена в директорию 'visualizations'.")
        else:
            print("Не найдено моделей с поддержкой извлечения важности признаков.")
    
    except Exception as e:
        print(f"Ошибка при создании визуализации важности признаков: {e}")

# Запуск всех функций визуализации
if __name__ == "__main__":
    print("Создание сравнительных визуализаций...")
    create_comparison_visualizations()
    
    print("\nСоздание визуализаций распределения ошибок...")
    create_error_distributions()
    
    print("\nСоздание визуализации важности признаков...")
    create_feature_importance()
    
    print("\nВсе визуализации созданы и сохранены в директорию 'visualizations'.")
