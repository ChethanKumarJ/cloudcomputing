import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.api.java.JavaRDD;
import scala.Tuple2;

public class WineQualityPrediction {
    public static void main(String[] args) {
        System.out.println("Starting Spark Application");
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityPrediction")
                .config("spark.some.config.option", "config-value")
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");
        String localPath = "ValidationDataset.csv";
        String trainedModelPath = "spark-model";
        String trainedModelOutputPath = "/opt/spark-model";

        try {
            System.out.println("Loading validation data from " + localPath);
            Dataset<Row> rawDataFrame = spark.read()
                    .format("csv")
                    .option("header", "true")
                    .option("sep", ";")
                    .option("inferschema", "true")
                    .load(localPath);

            Dataset<Row> cleanDataFrame = cleanData(rawDataFrame);
            System.out.println("Loading the trained model from " + trainedModelPath);
            PipelineModel predictionModel = PipelineModel.load(trainedModelPath);
            Dataset<Row> predictions = predictionModel.transform(cleanDataFrame);
            MulticlassClassificationEvaluator accuracyEvaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction")
                    .setMetricName("accuracy");
            double accuracy = accuracyEvaluator.evaluate(predictions);
            System.out.println("Test Accuracy of wine prediction model = " + accuracy);

            JavaRDD<Tuple2<Object, Object>> predictionResults = predictions.select("prediction", "label")
                    .javaRDD()
                    .map(row -> new Tuple2<>(row.getDouble(0), row.getDouble(1)));

            MulticlassMetrics predictionMetrics = new MulticlassMetrics(predictionResults.rdd());
            double weightedF1Score = predictionMetrics.weightedFMeasure();
            System.out.println("Weighted F1 Score of wine prediction model = " + weightedF1Score);
            System.out.println("Saving the trained model to " + trainedModelOutputPath);
            predictionModel.write().overwrite().save(trainedModelOutputPath);

        } catch (Exception e) {
            System.out.println("An error occurred: " + e.getMessage());
        }

        System.out.println("Exiting Spark Application");
        spark.stop();
    }

    private static Dataset<Row> cleanData(Dataset<Row> dataFrame) {
        for (String column : dataFrame.columns()) {
            dataFrame = dataFrame.withColumn(column.replace("\"", ""), dataFrame.col(column).cast("double"));
        }
        return dataFrame;
    }
}

