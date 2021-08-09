import org.apache.spark.sql.{DataFrame, SparkSession}

object Preprocessing {
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

}
