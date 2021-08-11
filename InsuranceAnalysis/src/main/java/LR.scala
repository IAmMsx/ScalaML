import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.Row

object LR {
    def main(args: Array[String]): Unit = {
        val spark = SparkSessionCreate.createSession()
        import spark.implicits._


        // 超参数
        val numFolds = 10 // 交叉验证折数
        val MaxIter: Seq[Int] = Seq(1000) // 最大迭代次数
        val RegParam = Seq(0.001) // 回归参数值
        val Tol = Seq(1e-6) // 容错值
        val ElasticNetParam = Seq(0.001) // 弹性网络参数

        // 创建LR估计器
        val model = new LinearRegression().setFeaturesCol("features").setLabelCol("label")

        println("Building ML pipeline") // 构建管道估计器
        val pipeline: Pipeline = new Pipeline()
          .setStages((Preprocessing.stringIndexerStages :+ Preprocessing.assembler) :+ model)

        // 构建参数网格
        val paramGrid = new ParamGridBuilder()
          .addGrid(model.maxIter, MaxIter)
          .addGrid(model.regParam, RegParam)
          .addGrid(model.tol, Tol)
          .addGrid(model.elasticNetParam, ElasticNetParam)
          .build()

        println("Preparing K-fold Cross Validation and Grid Search: Model tuning")
        // 构建交叉验证估算器
        val cv: CrossValidator = new CrossValidator()
          .setEstimator(pipeline)
          .setEvaluator(new RegressionEvaluator())
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(numFolds)

        println("Training model with Linear Regression algorithm")
        // 训练线性回归模型
        val cvModel: CrossValidatorModel = cv.fit(Preprocessing.trainingData)

        // save the workFlow
//        cvModel.write.overwrite().save("model/LR_model")
        // Load the workFlow back
//        val saveCV = CrossValidatorModel.load("model/LR_model")

        println("Evaluating model on train and validation set and calculating RMSE")
        val trainPredictionsAndLabels = cvModel.transform(Preprocessing.trainingData)
          .select("label", "prediction")
          .map { case Row(label: Double, prediction: Double) => (label, prediction) }
          .rdd

        val validPredictionsAndLabels = cvModel.transform(Preprocessing.validationData)
          .select("label", "prediction")
          .map { case Row(label: Double, prediction: Double) => (label, prediction) }
          .rdd

        // 计算训练集和测试集的原始预测
        val trainRegressionMetrics: RegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
        val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)
        // 接下来寻找最好的模型
        val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]

        val results = "\n=====================================================================\n" +
          s"Param trainSample: ${Preprocessing.trainSample}\n" +
          s"Param testSample: ${Preprocessing.testSample}\n" +
          s"TrainingData count: ${Preprocessing.trainingData.count}\n" +
          s"ValidationData count: ${Preprocessing.validationData.count}\n" +
          s"TestData count: ${Preprocessing.testData.count}\n" +
          "=====================================================================\n" +
          s"Param maxIter = ${MaxIter.mkString(",")}\n" +
          s"Param numFolds = $numFolds\n" +
          "=====================================================================\n" +
          s"Training data MSE = ${trainRegressionMetrics.meanSquaredError}\n" +
          s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
          s"Training data R-squared = ${trainRegressionMetrics.r2}\n" +
          s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError}\n" +
          s"Training data Explained Variance = ${trainRegressionMetrics.explainedVariance}\n" +
          "=====================================================================\n" +
          s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
          s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
          s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
          s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
          s"Validation data Explained Variance = ${validRegressionMetrics.explainedVariance}\n" +
          s"CV param explained: ${cvModel.explainParams}\n" +
          s"GBT params explained: ${bestModel.stages.last.asInstanceOf[LinearRegressionModel].explainParams}\n" +
          "=====================================================================\n"
        println(results)

        //
        println("Run prediction on the test set")
        cvModel.transform(Preprocessing.testData)
          .select("id","prediction")
          .withColumnRenamed("prediction","loss")
          .coalesce(1) // 获取单个csv文件中的全部预测
          .write.format("com.databricks.spark.csv")
          .option("header","true")
          .save("output/result_LR.csv")

        spark.stop()
    }

}
