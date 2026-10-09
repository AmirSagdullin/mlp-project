# Многослойный персептрон с нуля

Реализация многослойного персептрона на NumPy для задачи бинарной классификации
рака молочной железы (Breast Cancer Wisconsin Diagnostic).

## Структура проекта

- `data/data.csv` — исходный датасет (569 объектов, 30 признаков)
- `data_split/` — обучающая и валидационная выборки
- `src/layers.py` — реализация слоя DenseLayer и функций активации
- `src/model.py` — сборка сети, forward/backward, обучение, сохранение
- `src/split_data.py` — разделение данных со стратификацией
- `src/train.py` — обучение модели
- `src/predict.py` — предсказание на новых данных
- `notebooks/EDA.ipynb` — первичный анализ данных
- `report/report.tex` — отчёт по проекту

## Установка

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt