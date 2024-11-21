import sys
import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col

def clean_data(data_frame):
    """Cleans data by casting columns to double and stripping extra quotes."""
    return data_frame.select(*(col(column).cast("double").alias(column.strip('\"')) for column in data_frame.columns))

if __name__ == "__main__":
    print("Starting Spark Application")
    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # Configuration for using S3
    spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    training_data_path = "s3://path/to/TrainingDataset.csv"
    model_output_path = "s3://path/to/spark-model"

    try:
        print(f"Loading training data from {training_data_path}")
        raw_data_frame = spark.read.format("csv").option('header', 'true').option("sep", ";").option("inferschema", 'true').load(training_data_path)
        training_data_frame = clean_data(raw_data_frame)

        feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                           'pH', 'sulphates', 'alcohol']
        label_column = 'quality'

        assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
        indexer = StringIndexer(inputCol=label_column, outputCol="label")

        rf_classifier = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=100, maxDepth=10, seed=42)
        
        pipeline = Pipeline(stages=[assembler, indexer, rf_classifier])

        # Set up CrossValidator
        param_grid = ParamGridBuilder() \
            .addGrid(rf_classifier.maxDepth, [5, 10, 15]) \
            .addGrid(rf_classifier.numTrees, [50, 100, 150]) \
            .build()

        evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')

        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
        cv_model = cv.fit(training_data_frame)

        # Select the best model
        best_model = cv_model.bestModel
        print(f"Best model parameters: {best_model.stages[-1].extractParamMap()}")

        # Save the best model
        print(f"Saving the best model to {model_output_path}")
        best_model.write().overwrite().save(model_output_path)

    except Exception as e:
        print(f"An error occurred: {e}")

    spark.stop()
