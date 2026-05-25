from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("Cancer Diagnosis") \
    .getOrCreate()

# Load dataset
data = spark.read.csv("project3_data.csv", header=True, inferSchema=True)

# Drop ID column
data = data.drop("id")

# Encode label as B=0, M=1
indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
data = indexer.fit(data).transform(data)

# Assemble feature columns into single vector
feature = [c for c in data.columns if c not in ("diagnosis", "label")]
assembler = VectorAssembler(inputCols=feature, outputCol="features")
data = assembler.transform(data)

# Keep only features and label columns
data = data.select("features", "label")

# Split data into 80% training and 20% testing
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train Random Forest
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, seed=42)
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)

# Train Gradient Boosting
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=50, seed=42)
gbt_model = gbt.fit(train_data)
gbt_predictions = gbt_model.transform(test_data)

# Evaluate model
def evaluate(predictions):
    metrics = {}
    for name in ["f1", "weightedPrecision", "weightedRecall", "accuracy"]:
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=name)
        metrics[name] = evaluator.evaluate(predictions)
    return metrics

rf_metrics = evaluate(rf_predictions)
gbt_metrics = evaluate(gbt_predictions)


# Print result for both models
print("Random Forest Evaluation:")
for name, score in rf_metrics.items():
    if name == "accuracy":
        print(f"{name}: {score*100:.4f}%")
    else:
        print(f"{name}: {score:.4f}")

print("\nGradient Boosting Evaluation:")
for name, score in gbt_metrics.items():
    if name == "accuracy":
        print(f"{name}: {score*100:.4f}%")
    else:
        print(f"{name}: {score:.4f}")

