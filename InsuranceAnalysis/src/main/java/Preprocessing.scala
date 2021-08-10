import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Preprocessing {
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



}
