import Preproessing.trainInput
import org.apache.spark.sql.{DataFrame, SparkSession}

object EDA {
    def main(args: Array[String]): Unit = {
        val spark: SparkSession = SparkSessionCreate.createSession()
        import spark.implicits._

        val df: DataFrame = Preproessing.trainInput
        df.show()

        // df.printSchema() 显示dafaFrame的结构
//        println(df.printSchema())
//        df.printSchema()
//
//        println("*******************")
//        println(df.count())
//
//        df.select("id","cat1","cat2","cat3","cont2","cont3","loss")
//          .show()

        val newDF = df.withColumnRenamed("loss", "label")
        newDF.createOrReplaceTempView("insurance")

        spark.sql("SElECT avg(insurance.label) as AVG_LOSS FROM insurance").show()
        spark.sql("SElECT min(insurance.label) as MIN_LOSS FROM insurance").show()
        spark.sql("SElECT max(insurance.label) as MAX_LOSS FROM insurance").show()


    }

}
