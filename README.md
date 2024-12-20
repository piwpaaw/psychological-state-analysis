# Psychological State Analysis

## Описание
Этот проект выполняет анализ и предсказание психологического состояния на основе датасета. Используется PySpark для обработки данных и машинного обучения.

## Шаги выполнения:
1. **Загрузка данных.** Анализ структуры датасета.
2. **Анализ данных.** Проверка пропущенных значений и статистики.
3. **Предобработка.** Подготовка данных для модели (заполнение пропусков, преобразование категорий в числа).
4. **Обучение модели.** Логистическая регрессия для предсказания целевой переменной.
5. **Подбор гиперпараметров.** Улучшение точности модели с помощью кросс-валидации.

## Файлы
- `ioy.py` — основной код проекта.
- `output.txt` — вывод программы (результаты анализа, обучения, тестирования).
- `dataset.csv` — входной файл(dataset).

## Установка и запуск
1. Установите зависимости: `pyspark`.
2. Запустите проект: `ioy.py`.
