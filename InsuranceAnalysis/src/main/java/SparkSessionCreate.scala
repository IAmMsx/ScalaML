import org.apache.spark.sql.SparkSession

object SparkSessionCreate {
    def createSession(): SparkSession ={
        val spark = SparkSession
          .builder
          .master("local[*]")
          .config("spark.sql.warehouse.dir", "D:\\javaWork\\ScalaML\\InsuranceAnalysis")
          .appName("InsuranceAnalysis")
          .getOrCreate()
        spark
    }
}
