import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.Row

object GBT {
    def main(args: Array[String]): Unit = {
        val spark = SparkSessionCreate.createSession()

        import spark.implicits._

        // 参数设置
        val NumTrees = Seq(5, 10, 15) // 树的数量
        val MaxBins = Seq(5, 7, 9) // 最大分箱数
        val numFolds = 10 // 10折交叉验证
        val MaxIter = Seq(10) // 最大迭代次数
        val MaxDepth = Seq(10) // 树最大深度

        // 创建GBT估计器
        val model = new GBTRegressor().setFeaturesCol("features").setLabelCol("label")

        // 链接转换和预测器以构建管道
        val pipeline = new Pipeline().setStages((Preprocessing.stringIndexerStages :+ Preprocessing.assembler) :+ model)

        println("Preparing K-fold Cross Validation and Grid Search")
        // 构建参数网格
        val paramGrid = new ParamGridBuilder()
          .addGrid(model.maxIter, MaxIter)
          .addGrid(model.maxBins, MaxBins)
          .addGrid(model.maxDepth, MaxDepth)
          .build()

        // 构建交叉验证估算器
        val cv: CrossValidator = new CrossValidator()
          .setEstimator(pipeline)
          .setEvaluator(new RegressionEvaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(numFolds)

        println("Training model with GradientBoostedTrees algorithm")
        // 训练GBT模型
        val cvModel = cv.fit(Preprocessing.trainingData)

        println("Evaluating model on train and test data and calculating RMSE")
        val trainPredictionsAndLabels = cvModel.transform(Preprocessing.trainingData)
          .select("label", "prediction")
          .map { case Row(label: Double, prediction: Double) => (label, prediction) }
          .rdd

        val validPredictionsAndLabels = cvModel.transform(Preprocessing.validationData)
          .select("label", "prediction")
          .map { case Row(label: Double, prediction: Double) => (label, prediction) }
          .rdd

        // 计算训练集和验证集的原始预测
        val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
        val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)

        // 寻找最好模型
        val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]

        // 测量特征的重要性
        val featureImportances = bestModel.stages.last.asInstanceOf[GBTRegressionModel].featureImportances.toArray
        // 升序排序
        val FI_to_List_Sorted = featureImportances.toList.sorted.toArray
        println(FI_to_List_Sorted.mkString("Array(", ", ", ")"))

        val output = "\n=====================================================================\n" +
          s"Param trainSample:${Preprocessing.trainSample}\n" +
          s"Param testSample:${Preprocessing.testSample}\n" +
          s"TrainingData count:${Preprocessing.trainingData.count}\n" +
          s"validationData count:${Preprocessing.validationData.count}\n" +
          s"testData count:${Preprocessing.testData.count}\n" +
          "=====================================================================\n" +
          s"Param maxIter = ${MaxIter.mkString(",")}\n" +
          s"Param maxDepth = ${MaxDepth.mkString(",")}\n" +
          s"Param numFolds = $numFolds\n" +
          "=====================================================================\n" +
          s"Training Data MSE = ${trainRegressionMetrics.meanSquaredError}\n" +
          s"Training Data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
          s"Training Data R-squared = ${trainRegressionMetrics.r2}\n" +
          s"Training Data MAE = ${trainRegressionMetrics.meanAbsoluteError}\n" +
          s"Training Data Explained Variance = ${trainRegressionMetrics.explainedVariance}\n" +
          "=====================================================================\n" +
          s"validation Data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
          s"validation Data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
          s"validation Data R-squared = ${validRegressionMetrics.r2}\n" +
          s"validation Data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
          s"validation Data Explained Variance = ${validRegressionMetrics.explainedVariance}\n" +
          "=====================================================================\n" +
          s"CV param explained: ${cvModel.explainParams}\n" +
          s"GBT params explained: ${bestModel.stages.last.asInstanceOf[GBTRegressionModel].explainParams}\n" +
          s"GBT features importances:\n " +
          s"${Preprocessing.featureCols.zip(FI_to_List_Sorted).map(t => s"\t${t._1} = ${t._2}").mkString("\n")}\n" +
          "=====================================================================\n"

        println(output)

        println("Run prediction over test dataset")
        cvModel.transform(Preprocessing.testData)
          .select("id","prediction")
          .withColumnRenamed("prediction","loss")
          .coalesce(1)
          .write.format("csv")
          .option("header","true")
          .save("output/result_GBT.csv")

        spark.stop()

    }

}
