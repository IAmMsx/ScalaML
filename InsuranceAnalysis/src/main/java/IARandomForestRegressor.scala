import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.Row

object IARandomForestRegressor {
    def main(args: Array[String]): Unit = {
        val spark = SparkSessionCreate.createSession()
        import spark.implicits._

        // 定义超参数
        val NumTrees = Seq(5, 10, 15)
        val Maxbins = Seq(23, 27, 30)
        val numFolds = 5
        val MaxIter = Seq(20)
        val MaxDepth = Seq(20)

        val model = new RandomForestRegressor().setFeaturesCol("features").setLabelCol("label")
        println("Building Machine Learning pipeline")
        val pipeline = new Pipeline().setStages(
            (Preproessing.stringIndexerStages :+ Preproessing.assembler) :+ model
        )

        val paramGrid = new ParamGridBuilder()
          .addGrid(model.numTrees, NumTrees)
          .addGrid(model.maxBins, Maxbins)
          .addGrid(model.maxDepth, MaxDepth)
          .build()

        println("Preparing K-fold Cross Validation and Grid Search")
        val cv = new CrossValidator()
          .setEstimator(pipeline)
          .setEvaluator(new RegressionEvaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(numFolds)

        println("Training model with Random Forest algorithm")
        val cvModel = cv.fit(Preproessing.trainingData)

        println("Evaluating model on train and test data and calculating RMSE")
        val trainPredictionsAndLabels = cvModel.transform(Preproessing.trainingData).select("label", "prediction")
          .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

        val validPredictionsAndLabels = cvModel.transform(Preproessing.validationData).select("label", "prediction")
          .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

        val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
        val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)

        val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
        val featureImportances = bestModel.stages.last.asInstanceOf[RandomForestRegressionModel].featureImportances.toArray
        val FI_to_List_sorted = featureImportances.toList.sorted.toArray

        val output = "\n=====================================================================\n" +
          s"Param trainSample: ${Preproessing.trainSample}\n" +
          s"Param testSample: ${Preproessing.testSample}\n" +
          s"TrainingData count: ${Preproessing.trainingData.count}\n" +
          s"ValidationData count: ${Preproessing.validationData.count}\n" +
          s"TestData count: ${Preproessing.testData.count}\n" +
          "=====================================================================\n" +
          s"Param maxIter = ${MaxIter.mkString(",")}\n" +
          s"Param maxDepth = ${MaxDepth.mkString(",")}\n" +
          s"Param numFolds = $numFolds\n" +
          "=====================================================================\n" +
          s"Training data MSE = ${trainRegressionMetrics.meanSquaredError}\n" +
          s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
          s"Training data R-squared = ${trainRegressionMetrics.r2}\n" +
          s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError}\n" +
          s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance}\n" +
          "=====================================================================\n" +
          s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
          s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
          s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
          s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
          s"Validation data Explained variance = ${validRegressionMetrics.explainedVariance}\n" +
          "=====================================================================\n" +
          s"CV params explained: ${cvModel.explainParams}\n" +
          s"RF params explained: ${bestModel.stages.last.asInstanceOf[RandomForestRegressionModel].explainParams}\n" +
          s"RF features importances:\n ${Preproessing.featureCols.zip(FI_to_List_sorted).map(t => s"\t${t._1} = ${t._2}").mkString("\n")}\n" +
          "=====================================================================\n"

        println(output)

        println("Run prediction over test dataset")
        cvModel.transform(Preproessing.testData)
          .select("id","prediction")
          .withColumnRenamed("prediction","loss")
          .coalesce(1)
          .write.format("csv")
          .option("header","true")
          .save("output/result_RF.csv")


        spark.stop()
    }

}
