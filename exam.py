# Импорты
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count
from pyspark.sql.types import StringType
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Создание Spark-сессии
spark = SparkSession.builder \
    .appName("Full Lab Work: Psychological State Analysis") \
    .getOrCreate()

# Укажите путь к файлу
file_path = "C:/data/dataset.csv"

### Шаг 1: Загрузка данных
print("\nШаг 1: Загрузка данных\n")
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Сколько столбцов и строк в данных
print(f"Количество строк: {df.count()}, Количество столбцов: {len(df.columns)}\n")

# Показать первые 5 строк данных
df.show(5)

# Показать типы данных
print("\nТипы данных в столбцах:")
df.printSchema()

### Шаг 2: Анализ данных
print("\nШаг 2: Анализ данных\n")
# Разделение столбцов по типам
numerical_cols = [c[0] for c in df.dtypes if c[1] in ['int', 'double']]
non_numerical_cols = [c[0] for c in df.dtypes if c[1] not in ['int', 'double']]

# Пропущенные значения в числовых и нечисловых столбцах
print("\nПропущенные значения в числовых столбцах:")
df.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in numerical_cols]).show()

print("\nПропущенные значения в нечисловых столбцах:")
df.select([count(when(col(c).isNull(), c)).alias(c) for c in non_numerical_cols]).show()

# Описательная статистика
print("\nОписательная статистика для числовых столбцов:")
df.describe().show()

### Шаг 3: Предобработка данных
print("\nШаг 3: Предобработка данных\n")
# Заполнение пропусков в числовых данных
imputer = Imputer(inputCols=numerical_cols, outputCols=[f"{c}_imputed" for c in numerical_cols])
df = imputer.fit(df).transform(df)

# Преобразование Timestamp в строки
timestamp_cols = [c[0] for c in df.dtypes if c[1] == 'timestamp']
for col_name in timestamp_cols:
    df = df.withColumn(col_name, col(col_name).cast(StringType()))

# Преобразование категориальных данных в числовые
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed") for col in non_numerical_cols]
pipeline = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)

# Определяем X (features) и Y (Psychological State)
assembler = VectorAssembler(inputCols=[f"{c}_imputed" for c in numerical_cols] + 
                            [f"{col}_indexed" for col in non_numerical_cols if col != "Psychological State"], 
                            outputCol="features")
df = assembler.transform(df)
df.select("features", "Psychological State_indexed").show(df.count(), truncate=False)

### Шаг 4: Обучение модели
print("\nШаг 4: Обучение модели\n")
# Разделение на обучающую и тестовую выборки
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Обучение логистической регрессии
lr = LogisticRegression(featuresCol="features", labelCol="Psychological State_indexed", maxIter=10)
lr_model = lr.fit(train_df)

# Прогнозирование
predictions = lr_model.transform(test_df)
predictions.select("Psychological State_indexed", "prediction", "features").show(predictions.count(), truncate=False)

### Шаг 5: Подбор гиперпараметров
print("\nШаг 5: Подбор гиперпараметров\n")
# Настройка параметров для кросс-валидации
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = MulticlassClassificationEvaluator(labelCol="Psychological State_indexed", metricName="accuracy")
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Обучение с кросс-валидацией
cv_model = cv.fit(train_df)

# Показ лучших параметров
best_model = cv_model.bestModel
print("\nЛучшие параметры:")
for param, value in best_model.extractParamMap().items():
    print(f"{param.name}: {value}")

# Точность лучшей модели
final_predictions = best_model.transform(test_df)
final_accuracy = evaluator.evaluate(final_predictions)
print(f"\nТочность лучшей модели: {final_accuracy:.4f}")
