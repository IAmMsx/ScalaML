import Preprocessing.trainInput
import org.apache.spark.sql.{DataFrame, SparkSession}

object EDA {
    def main(args: Array[String]): Unit = {
        val spark: SparkSession = SparkSessionCreate.createSession()
        import spark.implicits._

        val df: DataFrame = Preprocessing.trainInput
        df.show()


    }

}
