import sys
import os

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def clean_data(data_frame):
    return data_frame.select(*(col(column).cast("double").alias(column.strip('\"')) for column in data_frame.columns))

if __name__ == "__main__":
    print("Starting Spark Application")

    # Setting up the Spark session
    spark = SparkSession.builder \
            .appName("WineQualityPrediction") \
            .config("spark.some.config.option", "config-value") \
            .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    # Setup paths
    local_path = "ValidationDataset.csv"  
    trained_model_path = "spark-model"  
    trained_model_output_path = "/opt/spark-model"  

    try:
        # Load and clean the validation dataset
        raw_data_frame = spark.read.format("csv").option('header', 'true').option("sep", ";").option("inferschema", 'true').load(local_path)
        clean_data_frame = clean_data(raw_data_frame)

        # Load the trained model
        prediction_model = PipelineModel.load(trained_model_path)
        predictions = prediction_model.transform(clean_data_frame)

        # Evaluate the model
        accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
        accuracy = accuracy_evaluator.evaluate(predictions)
        print(f'Test Accuracy of wine prediction model = {accuracy}')

        # Compute the F1 score
        prediction_results = predictions.select(['prediction', 'label'])
        prediction_metrics = MulticlassMetrics(prediction_results.rdd.map(tuple))
        weighted_f1_score = prediction_metrics.weightedFMeasure()
        print(f'Weighted F1 Score of wine prediction model = {weighted_f1_score}')

        # Save the trained model (Consider adjusting the path for S3 or other storage)
        prediction_model.write().overwrite().save(trained_model_output_path)

    except Exception as e:
        print(f"An error occurred: {e}")

    print("Exiting Spark Application")
    spark.stop()
