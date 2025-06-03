import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from catboost import CatBoostRegressor
from PIL import Image
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet

# Настройка страницы
st.set_page_config(
    page_title="ML Models Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Функция для загрузки моделей
@st.cache_resource
def load_models():
    models = {}
    # Загрузка модели ML1 (ElasticNet)
    try:
        models['ElasticNet'] = pickle.load(open('models/model_ml1.pkl', 'rb'))
    except:
        st.warning("Модель ElasticNet не найдена")
    
    # Загрузка модели ML2 (GradientBoosting)
    try:
        if 'model_ml2.pkl' in os.listdir('models'):
            file_size = os.path.getsize('models/model_ml2.pkl')
            st.info(f"Размер файла model_ml2.pkl: {file_size} байт")
            
            # Попытка загрузить модель с отладочной информацией
            try:
                import joblib
                # Сначала пробуем joblib, который более надежен при несовместимости версий
                models['GradientBoosting'] = joblib.load('models/model_ml2.pkl')
            except Exception as joblib_error:
                st.warning(f"Не удалось загрузить через joblib: {str(joblib_error)}")
                
                # Пробуем через pickle
                try:
                    with open('models/model_ml2.pkl', 'rb') as f:
                        models['GradientBoosting'] = pickle.load(f)
                except Exception as pickle_error:
                    st.error(f"Не удалось загрузить через pickle: {str(pickle_error)}")
                    
                    # Если не удалось загрузить, создаем новую модель
                    st.warning("🔄 Создание новой модели GradientBoosting...")
                    from sklearn.ensemble import GradientBoostingRegressor
                    
                    # Загружаем датасет и обучаем простую модель
                    try:
                        data = pd.read_csv('data/EDA_regression.csv')
                        X = data.drop('price', axis=1)
                        y = data['price']
                        
                        # Обучаем простую модель
                        simple_gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
                        simple_gb.fit(X, y)
                        models['GradientBoosting'] = simple_gb
                        st.success("✅ Создана новая модель GradientBoosting")
                    except Exception as train_error:
                        st.error(f"Не удалось создать новую модель: {str(train_error)}")
        else:
            st.warning("❌ Файл модели GradientBoosting (model_ml2.pkl) не найден")
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке модели GradientBoosting: {str(e)}")
        import traceback
        st.error(f"Трассировка ошибки: {traceback.format_exc()}")
    
    # Загрузка модели ML3 (CatBoost)
    try:
        cb_model = CatBoostRegressor()
        cb_model.load_model('models/model_ml3.cbm')
        models['CatBoost'] = cb_model
    except:
        st.warning("Модель CatBoost не найдена")
    
    # Загрузка модели ML4 (Bagging)
    try:
        models['Bagging'] = pickle.load(open('models/model_ml4.pkl', 'rb'))
    except:
        st.warning("Модель Bagging не найдена")
    
    # Загрузка модели ML5 (Stacking)
    try:
        models['Stacking'] = pickle.load(open('models/model_ml5.pkl', 'rb'))
    except:
        st.warning("Модель Stacking не найдена")
    
    return models

# Функция для загрузки данных
@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/EDA_regression.csv')
    except:
        st.error("Файл с данными не найден. Убедитесь, что файл 'data/EDA_regression.csv' существует.")
        return None

# Сайдбар с навигацией
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите страницу:",
    ["О разработчике", "О наборе данных", "Визуализации", "Предсказания"]
)

# Загрузка данных и моделей
data = load_data()
models = load_models()

# Страница 1: О разработчике
if page == "О разработчике":
    st.title("О разработчике")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Попытка загрузить фото
        try:
            image = Image.open('developer_photo.jpg')
            st.image(image, caption="Фотография разработчика")
        except:
            st.info("Фотография разработчика отсутствует. Для добавления поместите фото 'developer_photo.jpg' в корневую директорию проекта.")
    
    with col2:
        st.markdown("""
        ### ФИО
        Фельде Сергей Дмитриевич
        
        ### Номер учебной группы
        ФИТ-231
        
        ### Тема РГР
        Разработка Web-приложения (дашборда)
        для инференса моделей ML и анализа данных

        """)

# Страница 2: О наборе данных
elif page == "О наборе данных":
    st.title("О наборе данных")
    
    if data is not None:
        st.header("Описание предметной области")
        st.markdown("""
        Данный набор данных содержит информацию о ценах бриллиантов. Целевой переменной является `price` - стоимость бриллианта.
        
        ### Особенности предобработки данных
        
        Предварительная обработка данных включала кодирование категориальных переменных
        
        ### Описание признаков
        """)
        st.markdown("""
        - `price` — цена в долларах США
        - `carat` — вес бриллианта в каратах
        - `cut` — качество огранки
        - `color` — цвет бриллианта
        - `clarity` — показатель прозрачности бриллианта
        - `x` — длина в мм
        - `y` — ширина в мм
        - `z` — глубина в мм
        - `depth` — общая глубина в процентах z / mean(x, y) = 2 * z / (x + y)
        - `table` — ширина вершины алмаза относительно самой широкой точки
        """)
        # Создание таблицы с описанием признаков
        features_description = pd.DataFrame({
            'Признак': data.columns,
            'Тип данных': data.dtypes.astype(str),
            'Количество значений': data.count().values,
            'Среднее значение': [round(data[col].mean(), 2) if pd.api.types.is_numeric_dtype(data[col]) else '-' for col in data.columns],
            'Минимум': [round(data[col].min(), 2) if pd.api.types.is_numeric_dtype(data[col]) else '-' for col in data.columns],
            'Максимум': [round(data[col].max(), 2) if pd.api.types.is_numeric_dtype(data[col]) else '-' for col in data.columns]
        })
        
        st.dataframe(features_description)
        
        # Общая статистика по набору данных
        st.header("Общая статистика")
        st.markdown(f"""
        - **Количество объектов**: {data.shape[0]}
        - **Количество признаков**: {data.shape[1] - 1} (не считая целевую переменную)
        - **Размер датасета**: {data.memory_usage().sum() / 1024 / 1024:.2f} МБ
        """)
        
        # Просмотр данных
        st.header("Просмотр данных")
        st.dataframe(data.head())

# Страница 3: Визуализации
elif page == "Визуализации":
    st.title("Визуализации данных")
    
    if data is not None:
        # Настройка стиля графиков
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette('Set2')
        
        st.header("Распределение целевой переменной (price)")
        
        # Визуализация 1: Распределение целевой переменной
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['price'], kde=True, ax=ax)
        ax.set_title('Распределение цен')
        ax.set_xlabel('Цена')
        ax.set_ylabel('Частота')
        st.pyplot(fig)
        
        # Визуализация 2: Диаграмма рассеяния и линия регрессии
        st.header("Взаимосвязь между признаками и целевой переменной")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('price') if 'price' in numeric_cols else None
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.selectbox('Выберите первый признак для визуализации:', numeric_cols, index=0)
        
        with col2:
            feature2 = st.selectbox('Выберите второй признак для визуализации:', numeric_cols, index=min(1, len(numeric_cols)-1))
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot для первого признака
        sns.regplot(x=feature1, y='price', data=data, ax=ax[0])
        ax[0].set_title(f'Зависимость цены от {feature1}')
        ax[0].set_xlabel(feature1)
        ax[0].set_ylabel('Цена')
        
        # Scatter plot для второго признака
        sns.regplot(x=feature2, y='price', data=data, ax=ax[1])
        ax[1].set_title(f'Зависимость цены от {feature2}')
        ax[1].set_xlabel(feature2)
        ax[1].set_ylabel('Цена')
        
        st.pyplot(fig)
        
        # Визуализация 3: Тепловая карта корреляций
        st.header("Тепловая карта корреляций")
        
        corr = data.select_dtypes(include=[np.number]).corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Корреляция между признаками')
        st.pyplot(fig)
        
        # Визуализация 4: Ящик с усами для числовых признаков
        st.header("Распределение числовых признаков (Box Plot)")
        
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.boxplot(data=data[numeric_cols], ax=ax)
        ax.set_title('Распределение числовых признаков')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        st.pyplot(fig)
        
        # Дополнительная визуализация: Парные графики
        st.header("Парные графики (Pairplot)")
        
        # Выбор признаков для парного графика
        selected_features = st.multiselect(
            'Выберите признаки для парного графика (рекомендуется не более 5):',
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))]
        )
        
        if len(selected_features) > 0:
            selected_features.append('price')
            
            # Ограничение количества строк для улучшения производительности
            sample_size = min(1000, len(data))
            data_sample = data.sample(sample_size, random_state=42)
            
            fig = sns.pairplot(data_sample[selected_features], diag_kind='kde', height=2.5)
            fig.fig.suptitle('Парные графики выбранных признаков', y=1.02)
            st.pyplot(fig.fig)
        else:
            st.warning("Выберите хотя бы один признак для построения парного графика.")

# Страница 4: Предсказания
elif page == "Предсказания":
    st.title("Получение предсказаний моделей ML")
    st.markdown("""
    ### Описание проекта
    
    В рамках данного проекта было реализовано 5 различных моделей машинного обучения для задачи регрессии:
    
    1. **ElasticNet** - линейная регрессия с регуляризацией L1 и L2
    2. **GradientBoostingRegressor** - ансамблевый метод, использующий градиентный бустинг
    3. **CatBoostRegressor** - ансамблевый метод, использующий продвинутый градиентный бустинг
    4. **BaggingRegressor** - ансамблевый метод, использующий бэггинг
    5. **StackingRegressor** - ансамблевый метод, объединяющий предсказания базовых моделей
    """)
    if data is not None and models:
        st.header("Выберите способ ввода данных")
        
        input_method = st.radio(
            "Метод ввода данных:",
            ["Ручной ввод", "Загрузка CSV файла"]
        )
        
        # Выбор модели для предсказания
        model_name = st.selectbox(
            "Выберите модель для предсказания:",
            list(models.keys())
        )
        
        # Получение признаков (без целевой переменной)
        features = data.drop('price', axis=1).columns.tolist()
        
        if input_method == "Ручной ввод":
            st.header("Введите значения признаков")
            
            # Создание словаря для хранения значений признаков
            input_values = {}
            
            # Создание формы для ввода значений
            col1, col2 = st.columns(2)
            
            # Распределение признаков по двум колонкам
            half_features = len(features) // 2
            
            for i, feature in enumerate(features):
                # Определение типа признака
                if feature not in ['cut', 'color', 'clarity']:
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    mean_val = float(data[feature].mean())
                    
                    # Определение шага в зависимости от диапазона значений 
                    step = (max_val - min_val) / 100.0
                    
                    # Выбор подходящего виджета в зависимости от диапазона значений
                    if i < half_features:
                        with col1:
                            if max_val - min_val <= 10:  # Небольшой диапазон
                                input_values[feature] = st.slider(
                                    f"{feature}:", min_val, max_val, mean_val, step=step
                                )
                            else:  # Большой диапазон
                                input_values[feature] = st.number_input(
                                    f"{feature}:", min_val, max_val, mean_val, step=step
                                )
                    else:
                        with col2:
                            if max_val - min_val <= 10:  # Небольшой диапазон
                                input_values[feature] = st.slider(
                                    f"{feature}:", min_val, max_val, mean_val, step=step
                                )
                            else:  # Большой диапазон
                                input_values[feature] = st.number_input(
                                    f"{feature}:", min_val, max_val, mean_val, step=step
                                )
                else:
                    # Для категориальных признаков (cut, color, clarity) используем целые числа
                    if feature in ['cut', 'color', 'clarity']:
                        # Для категориальных признаков задаем диапазон целых чисел
                        if feature == 'cut':
                            options = list(range(1, 6))  # Целые числа от 1 до 5
                            default_idx = 2  # Индекс значения 3
                        elif feature == 'color':
                            options = list(range(1, 8))  # Целые числа от 1 до 7
                            default_idx = 3  # Индекс значения 4
                        elif feature == 'clarity':
                            options = list(range(1, 9))  # Целые числа от 1 до 8
                            default_idx = 3  # Индекс значения 4
                        
                        if i < half_features:
                            with col1:
                                input_values[feature] = st.selectbox(
                                    f"{feature} (больше - лучше):",
                                    options=options,
                                    index=default_idx
                                )
                        else:
                            with col2:
                                input_values[feature] = st.selectbox(
                                    f"{feature} (больше - лучше):",
                                    options=options,
                                    index=default_idx
                                )
                    else:
                        # Для других категориальных признаков (если есть)
                        unique_values = data[feature].unique().tolist()
                        
                        if i < half_features:
                            with col1:
                                input_values[feature] = st.selectbox(
                                    f"{feature}:", unique_values
                                )
                        else:
                            with col2:
                                input_values[feature] = st.selectbox(
                                    f"{feature}:", unique_values
                                )
            
            # Создание DataFrame из введенных значений
            input_df = pd.DataFrame([input_values])
            
        else:  # Загрузка CSV файла
            st.header("Загрузите CSV файл с данными")
            st.markdown("""
            CSV файл должен содержать те же признаки, что и обучающий набор данных (без целевой переменной).
            """)
            
            uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    input_df = pd.read_csv(uploaded_file)
                    
                    # Проверка наличия всех необходимых признаков
                    missing_features = set(features) - set(input_df.columns)
                    if missing_features:
                        st.error(f"В загруженном файле отсутствуют следующие признаки: {', '.join(missing_features)}")
                        input_df = None
                    else:
                        # Отображение загруженных данных
                        st.subheader("Загруженные данные:")
                        st.dataframe(input_df.head())
                except Exception as e:
                    st.error(f"Ошибка при загрузке файла: {str(e)}")
                    input_df = None
            else:
                input_df = None
        
        # Кнопка для получения предсказания
        if input_df is not None and st.button("Получить предсказание"):
            try:
                # Получение предсказания выбранной модели
                model = models[model_name]
                prediction = model.predict(input_df)
                
                st.header("Результат предсказания")
                
                # Вывод предсказания для каждой строки данных
                if len(prediction) == 1:
                    st.success(f"Предсказанная цена: {prediction[0]:,.2f} USD")
                    st.info("Примечание: Предсказание является точечной оценкой. Фактическое значение может отличаться.")
                else:
                    # Для нескольких предсказаний создаем таблицу
                    result_df = pd.DataFrame({
                        'Номер строки': range(1, len(prediction) + 1),
                        'Предсказанная цена': [f"{price:,.2f} USD" for price in prediction]
                    })
                    st.dataframe(result_df)
                    
                    # Построение гистограммы предсказаний
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(prediction, kde=True, ax=ax)
                    ax.set_title('Распределение предсказанных цен')
                    ax.set_xlabel('Предсказанная цена')
                    ax.set_ylabel('Частота')
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Ошибка при получении предсказания: {str(e)}")
    else:
        if data is None:
            st.error("Невозможно получить предсказания: данные не загружены.")
        if not models:
            st.error("Невозможно получить предсказания: модели не загружены.")

# Подвал приложения
st.markdown("---")
st.markdown("© 2025 ML Models Dashboard | Расчетно-графическая работа | ОМГТУ")
