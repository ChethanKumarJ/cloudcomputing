import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
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
        String trainingDataPath = "TrainingDataset.csv";
        String modelOutputPath = "spark-model";

        try {
            System.out.println("Loading training data from " + trainingDataPath);
            Dataset<Row> rawDataFrame = spark.read()
                    .format("csv")
                    .option("header", "true")
                    .option("sep", ";")
                    .option("inferschema", "true")
                    .load(trainingDataPath);

            Dataset<Row> trainingDataFrame = cleanData(rawDataFrame);
            String[] featureColumns = new String[]{"fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"};
            String labelColumn = "quality";
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureColumns)
                    .setOutputCol("features");
            StringIndexer indexer = new StringIndexer()
                    .setInputCol(labelColumn)
                    .setOutputCol("label");
            RandomForestClassifier rfClassifier = new RandomForestClassifier()
                    .setLabelCol("label")
                    .setFeaturesCol("features")
                    .setNumTrees(100)
                    .setMaxDepth(10)
                    .setSeed(42);

            Pipeline pipeline = new Pipeline()
                    .setStages(new org.apache.spark.ml.PipelineStage[]{assembler, indexer, rfClassifier});
            ParamGridBuilder paramGridBuilder = new ParamGridBuilder()
                    .addGrid(rfClassifier.maxDepth(), new int[]{5, 10, 15})
                    .addGrid(rfClassifier.numTrees(), new int[]{50, 100, 150});
            org.apache.spark.ml.tuning.ParamMap[] paramGrid = paramGridBuilder.build();

            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction")
                    .setMetricName("accuracy");

            CrossValidator cv = new CrossValidator()
                    .setEstimator(pipeline)
                    .setEstimatorParamMaps(paramGrid)
                    .setEvaluator(evaluator)
                    .setNumFolds(5);

            PipelineModel cvModel = cv.fit(trainingDataFrame);
            PipelineModel bestModel = (PipelineModel) cvModel.bestModel();
            System.out.println("Best model parameters: " + bestModel.stages()[bestModel.stages().length - 1].extractParamMap());
            System.out.println("Saving the best model to " + modelOutputPath);
            bestModel.write().overwrite().save(modelOutputPath);

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

