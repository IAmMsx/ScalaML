import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object SparkSessionCreate {
    def createSession(): SparkSession ={
        val sparkConf = new SparkConf().setMaster("local[*]")
          .setAppName("InsuranceAnalysis")
          .set("spark.sql.broadcastTimeout", "1200")
          .set("spark.sql.autoBroadcastJoinThreshold","-1")
          .set("SPARK_EXECUTOR_MEMORY","3000M")
          .set("SPARK_DRIVER_MEMORY","4000M")

        val spark = SparkSession.builder().config(sparkConf).getOrCreate()
//        val spark = SparkSession
//          .builder
//          .master("local[*]")
//          .config("spark.sql.warehouse.dir", "D:\\javaWork\\ScalaML\\InsuranceAnalysis")
//          .appName("InsuranceAnalysis")
//          .getOrCreate()
        spark
    }
}
