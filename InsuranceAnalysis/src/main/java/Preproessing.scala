import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Preproessing {
    val trainSample = 1.0
    val testSample = 1.0
    val train = "data/train.csv"
    val test = "data/test.csv"

    val spark: SparkSession = SparkSessionCreate.createSession()
    import spark.implicits._
    println("Reading data from " + train + " file")

    val trainInput: DataFrame = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load(train)
      .cache()

    val testInput: DataFrame = spark.read
      .option("header", "true")
      .option("inferSchema", "true") //避免将浮点值视为字符串
      .format("csv")
      .load(test)
      .cache()

    println("Preparing data for training model")
    var data: Dataset[Row] = trainInput.withColumnRenamed("loss", "label")
      .sample(withReplacement = false, trainSample)

    // null check
    val DF: DataFrame = data.na.drop()
    if (DF == data)
        println("No null values in the DataFrame")
    else{
        println("Null values exist in the DataFrame")
        data = DF
    }
    val seed = 12345L
    val splits: Array[Dataset[Row]] = data.randomSplit(Array(0.75,0.25),seed)
    val (trainingData,validationData) = (splits(0),splits(1))

    trainingData.cache()
    validationData.cache()


    val testData: Dataset[Row] = testInput.sample(withReplacement = false, testSample).cache()

    // 识别分类列(数据集分 分类列与数据列)
    def isCateg(c: String): Boolean = c.startsWith("cat")

    def categNewCol(c: String): String = if (isCateg(c)) s"idx_$c" else c

    // 删除太多类别的列
    def removeTooManyCategs(c: String): Boolean = !(c matches "cat(109$|110$|112$|113$|116$)")

    // 选择特征列 删除Label与ID列
    def onlyFeatureCols(c: String): Boolean = !(c matches "id|label")

    val featureCols: Array[String] = trainingData.columns
      .filter(removeTooManyCategs)
      .filter(onlyFeatureCols)
      .map(categNewCol)

    // 不太懂 *****
    val stringIndexerStages: Array[StringIndexerModel] = trainingData.columns.filter(isCateg)
      .map(c => new StringIndexer()
        .setInputCol(c) // 设置输入列
        .setOutputCol(categNewCol(c)) //设置输出列
        .fit(trainInput.select(c).union(testInput.select(c)))
      )

    // 将给定的列转换为单个向量列
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")


}
