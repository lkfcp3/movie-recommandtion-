import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.SparkSession
import scala.math.sqrt
import org.jblas.DoubleMatrix
import org.apache.log4j.{Level, Logger}
import java.util.List  
import java.util.ArrayList  

case class UsersMovieName(user: Integer, top5movie: Array[String])

object movieRecommender {
  private val minSimilarity = 0.6

  def cosineSimilarity(vector1: DoubleMatrix, vector2: DoubleMatrix): Double = {
    vector1.dot(vector2) / (vector1.norm2() * vector2.norm2())
  }

  def computeSimilarity(model: MatrixFactorizationModel, dataDir: String, dateStr: String): Unit = {
    //calculate all the similarity and store the stuff whose sim > 0.5 to Redis.
    val productsVectorRdd = model.productFeatures
      .map{case (movieId, factor) =>
      val factorVector = new DoubleMatrix(factor)
      (movieId, factorVector)
    }
    
    val productsSimilarity = productsVectorRdd.cartesian(productsVectorRdd)
      .filter{ case ((movieId1, vector1), (movieId2, vector2)) => movieId1 != movieId2 }
      .map{case ((movieId1, vector1), (movieId2, vector2)) =>
        val sim = cosineSimilarity(vector1, vector2)
        (movieId1, movieId2, sim)
      }.filter(_._3 >= minSimilarity)
    
    productsSimilarity.map{ case (movieId1, movieId2, sim) => 
      movieId1.toString + "," + movieId2.toString + "," + sim.toString
    }.saveAsTextFile(dataDir + "allSimilarity_" + dateStr)

    productsVectorRdd.unpersist()
    productsSimilarity.unpersist()
  }



  def main(args: Array[String]) {

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("alsBatchRecommender").set("spark.dynamicAllocation.enabled","true").set("spark.executor.memory", "6g").set("spark.cores.max", "4")    
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    if (args.length < 1) {
        println("USAGE:")
        println("spark-submit --class \"movieRecommender\" xxx.jar Date_String")
        println("spark-submit --class \"movieRecommender\" xxx.jar 20170401")
        sys.exit()
    }
    val dateStr = args(0)

    //val dataDir = "hdfs://localhost:9000/user/hadoop/ml-100k/ml-100k/"
    val dataDir = "s3://group16-spark-project/spark/ml-20m/"

    //read data
    val movielens = sc.textFile(dataDir+"ratings.csv").mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    val clean_data = movielens.map(_.split(","))

    val rate = clean_data.map{case Array(user, item, rate, time) => ( rate.toDouble)} 
    val users = clean_data.map{case Array(user, item, rate, time) => ( user.toInt)} 
    val items = clean_data.map{case Array(user, item, rate, time) => ( item.toInt)} 
    //show the data size and distinct user number
    println(s"distinct user count ${users.distinct.count()}")
    println(s"distinct item count ${items.distinct.count()}")

    //split the training data and testingData
    val rating = clean_data.map{case Array(user, item, rate, time) => Rating( user.toInt, item.toInt, rate.toDouble)}
    val Array(trainData, testData) = rating.randomSplit(Array(0.7,0.3),7856)
    rating.first()


    val (rank, iterations, lambda) = (50, 10, 0.01)
    val model = MatrixFactorizationModel.load(sc,dataDir+"ALSmodel_201111")
    //val (rank, iterations, lambda) = findBestPrameter(trainData, rating)
    //val model = ALS.train(trainData, rank, iterations, lambda)
    //val model = ALS.train(rating, rank, numIterations,0.01,-1)

    trainData.unpersist()

    //computeSimilarity(model, dataDir, dateStr) //save cos sim.
    model.save(sc, dataDir + "ALSmodel_" + dateStr) //save model.

    val rmse = computeRmse(model, rating)
    println("the Rmse = " + rmse)

    //Read Item Information
    val ItemData = sc.textFile(dataDir+"movies.csv").mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }.map(_.split(",") match {case line => (line(0).toInt,line(1))})
    println(s"distinct item count ${ItemData.distinct.count()}")
    //recommendforusers
    // recommendforusers(model,ItemData,sc,dataDir,dateStr)

    sc.stop()
  }

  def recommendforusers(model: MatrixFactorizationModel, ItemData:RDD[(Int,String)], sc:SparkContext, dataDir: String, dateStr: String) :Unit = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    // val spark = SparkSession.builder().config("spark.some.config.option", "some-value").getOrCreate()
    // import spark.implicits._

    val users_recommender = model.recommendProductsForUsers(5).sortByKey(true)
    
    val users_recommendProduct = users_recommender.map(r => (r._1, r._2.map(c => c.product)))

    val ItemDataArray = ItemData.values.collect()
    val users_movie = users_recommendProduct.map(r => (r._1,r._2.map(x => ItemDataArray(x-1))))
    val users_movieDF = users_movie.map(attributes=>UsersMovieName(attributes._1, attributes._2)).toDF()

    users_movieDF.show()

  }
  def findBestPrameter(trainData: RDD[Rating], realRatings: RDD[Rating]): (Int, Int, Double) = {
    val evaluations =
      for (rank   <- Array(10, 50);
           iterations <- Array(5, 10);
           lambda <- Array(1.0,  0.001))
        yield {
          val model = ALS.train(trainData, rank, iterations, lambda)
          val rmse = computeRmse(model, realRatings)
          unpersist(model)
		  println("rank = " + rank + ",lambda = " + lambda + ",iterations = " + iterations + ",RMSE" + rmse)
          ((rank, iterations, lambda), rmse)
        }
    val ((rank, iterations, lambda), rmse) = evaluations.sortBy(_._2).head
    println("After parameter adjust, the best rmse = " + rmse )
    println("rank = " + rank + ",lambda = " + lambda + ",iterations = " + iterations)
    (rank, iterations, lambda)
  }

  def computeRmse(model: MatrixFactorizationModel, ratings: RDD[Rating]): Double = {
    val usersProducts = ratings.map{ case Rating(user, product, rate) =>
      (user, product)
    }

    val prediction = model.predict(usersProducts).map{ case Rating(user, product, rate) =>
      ((user, product), rate)
    }

    val realPredict = ratings.map{case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(prediction)

    sqrt(realPredict.map{ case ((user, product), (rate1, rate2)) =>
      val err = rate1 - rate2
      err * err
    }.mean())//mean = sum(list) / len(list)
  }

  def unpersist(model: MatrixFactorizationModel): Unit = {
    model.userFeatures.unpersist()
    model.productFeatures.unpersist()
  }
}
