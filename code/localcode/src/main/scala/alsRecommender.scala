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
case class Ratings(userId: Int, movieId: Int, rating: Double)
case class Movies(id: Int, name: String)
case class Users(id: Int, age: Int, occupation: String, cluster: Int)

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

    val conf = new SparkConf().setAppName("alsBatchRecommender").set("spark.executor.memory", "4g")
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
    val dataDir = "file:///home/hadoop/spark/ml-100k/"


    //read data
    val movielens = sc.textFile(dataDir+"u.data")
    val clean_data = movielens.map(_.split("\t"))
    val rate = clean_data.map{case Array(user, item, rate, time) => ( rate.toDouble)} 
    val users = clean_data.map{case Array(user, item, rate, time) => ( user.toInt)} 
    val items = clean_data.map{case Array(user, item, rate, time) => ( item.toInt)} 
    //show the data size and distinct user number
    println(s"distinct user count ${users.distinct.count()}")
    println(s"distinct item count ${items.distinct.count()}")

    //get the averaga scores for this moviedata
    val average_rating_data = clean_data.map{case Array(user, item, rate, time) => (item.toInt, rate.toDouble)}.reduceByKey(_+_)
    println(s"distinct item count ${average_rating_data.first()}")


    //split the training data and testingData
    val rating = clean_data.map{case Array(user, item, rate, time) => Rating( user.toInt, item.toInt, rate.toDouble)}
    val Array(trainData, testData) = rating.randomSplit(Array(0.7,0.3),7856)
    rating.first()

    //compute model
    // val (rank, iterations, lambda) = findBestPrameter(trainData, rating)
    // val model = ALS.train(trainData, rank, iterations, lambda)
    // trainData.unpersist()
    // computeSimilarity(model, dataDir, dateStr) //save cos sim.
    // model.save(sc, dataDir + "ALSmodel_" + dateStr) //save model.

    //load model directly
    val (rank, iterations, lambda) = (10, 10,  0.01)
    val model = MatrixFactorizationModel.load(sc,dataDir+"ALSmodel_1")

    //compute Rmse for this model
    val rmse = computeRmse(model, rating)
    println("the Rmse = " + rmse)


    //Read Item Information
    val movies = sc.textFile(dataDir+"u.item")
    val ItemData = movies.map(_.split("\\|") match {case line => (line(0).toInt,line(1))})
    println(s"distinct user count ${ItemData.distinct.count()}")
    //recommendforusers
    recommendforusers(model,ItemData,sc,dataDir,dateStr)

    //Kmeans for user and product
    val genres = sc.textFile(dataDir+"u.genre")
    genres.take(5).foreach(println)
    val genreMap = genres.filter(!_.isEmpty).map(line => line.
    split("\\|")).map(array => (array(1), array(0))).collectAsMap
    println(genreMap)
    UseKmeans(model,movies,genreMap)

    recommendFriends(model,rating,ItemData,sc,dataDir)

    sc.stop()
  }

  def recommendFriends(model: MatrixFactorizationModel,rating:RDD[Rating],ItemData:RDD[(Int,String)], sc:SparkContext, dataDir: String):Unit ={
    // define Euclidean distance function
    import org.apache.spark.mllib.clustering.KMeans
    import breeze.linalg._
    import breeze.numerics.pow
    def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]): Double = pow(v1 - v2, 2).sum

    import org.apache.spark.mllib.linalg.Vectors
    val userFactors = model.userFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
    val userVectors = userFactors.map(_._2)

    val numClusters = 5
    val numIterations = 10
    val numRuns = 3
    // train user model
    val userClusterModel = KMeans.train(userVectors, numClusters, numIterations, numRuns)

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    //将电影，用户，评分数据转换成为DataFrame，进行SparkSQL操作
    val movies = ItemData.map(m => Movies(m._1.toInt, m._2)).toDF()
    val ratings = rating.map(r=>Ratings(r.user,r.product,r.rating)).toDF()

    //user Information
    val userData = sc.textFile(dataDir+"u.user")  
    val userBasic = userData.map(_.split("\\|")match{
      case Array(id, age, sex, job, x) => (id.toInt,(age.toInt,job))
    })
    //println(userBasic.first())
    val usersWithFactors = userBasic.join(userFactors)
    val usersAssigned = usersWithFactors.map { case (id, ((age, job), vector)) => 
      val pred = userClusterModel.predict(vector)
      val clusterCentre = userClusterModel.clusterCenters(pred)
      val dist = computeDistance(DenseVector(clusterCentre.toArray), DenseVector(vector.toArray))
      (id, age, job, pred, dist) 
    }
    val users = usersAssigned.map{case (id, age, job, cluster, dist) => Users(id,age,job,cluster)}.toDF()

    //recommend results
    val newDF = ratings.filter(ratings("rating") >= 5)//filter movie rate 5
      .join(movies, ratings("movieId") === movies("id"))//join for movie DF
      .join(users, ratings("userId") === users("id"))//join for user DF
    //newDF.show()
    println("you may be interested in these people with userID: ")
    newDF.filter(users("age") <= 35)//filet the age range
    .filter(users("age") >= 15)
    .filter(users("occupation") === "programmer")//filter the jobid
    .filter(movies("id") === 1) //filter the movie you like
    .select(users("id"))
    .take(10)
    .foreach(println)


  }


  def UseKmeans(model: MatrixFactorizationModel, movies:RDD[String], genreMap: scala.collection.Map[String,String]):Unit = {
    //generate Kmeans model features for user and movies
    import org.apache.spark.mllib.linalg.Vectors
    val movieFactors = model.productFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
    val movieVectors = movieFactors.map(_._2)
    val userFactors = model.userFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
    val userVectors = userFactors.map(_._2)

    // train models
    import org.apache.spark.mllib.clustering.KMeans
    val numClusters = 5
    val numIterations = 10
    val numRuns = 3
    val movieClusterModel = KMeans.train(movieVectors, numClusters, numIterations, numRuns)
    val userClusterModel = KMeans.train(userVectors, numClusters, numIterations, numRuns)
    println("We total set 5 clusters")

    //prredict movie cluster for first movie
    val movie1 = movieVectors.first
    val movie1ID = movieFactors.first._1
    val movieCluster = movieClusterModel.predict(movie1)
    println(s"${movie1ID} belong to cluster ${movieCluster}")


    val titlesAndGenres = movies.map(_.split("\\|")).map { array =>
    val genres = array.toSeq.slice(5, array.size)
    val genresAssigned = genres.zipWithIndex.filter { case (g, idx) => g == "1"}.map { case (g, idx) =>
    genreMap(idx.toString)
    }
    (array(0).toInt, (array(1), genresAssigned))
    }
    println(titlesAndGenres.first)

    // define Euclidean distance function
    import breeze.linalg._
    import breeze.numerics.pow
    def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]): Double = pow(v1 - v2, 2).sum

    // join titles with the factor vectors, and compute the distance of each vector from the assigned cluster center
    val titlesWithFactors = titlesAndGenres.join(movieFactors)
    val moviesAssigned = titlesWithFactors.map { case (id, ((title, genres), vector)) => 
      val pred = movieClusterModel.predict(vector)
      val clusterCentre = movieClusterModel.clusterCenters(pred)
      val dist = computeDistance(DenseVector(clusterCentre.toArray), DenseVector(vector.toArray))
      (id, title, genres.mkString(" "), pred, dist) 
    }
    val clusterAssignments = moviesAssigned.groupBy { case (id, title, genres, cluster, dist) => cluster }.collectAsMap 

    for ( (k, v) <- clusterAssignments.toSeq.sortBy(_._1)) {
      println(s"Cluster $k:")
      val m = v.toSeq.sortBy(_._5)
      println(m.take(10).map { case (_, title, genres, _, d) => (title, genres, d) }.mkString("\n")) 
      println("=====\n")
    }


    // compute the cost (WCSS) on for movie and user clustering
    val movieCost = movieClusterModel.computeCost(movieVectors)
    val userCost = userClusterModel.computeCost(userVectors)
    println("WCSS for movies: " + movieCost)
    println("WCSS for users: " + userCost)

  }

  def recommendforusers(model: MatrixFactorizationModel, ItemData:RDD[(Int,String)], sc:SparkContext, dataDir: String, dateStr: String) :Unit = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val users_recommender = model.recommendProductsForUsers(5).sortByKey(true)
    val users_recommendProduct = users_recommender.map(r => (r._1, r._2.map(c => c.product)))
    val ItemDataArray = ItemData.values.collect()
    val users_movie = users_recommendProduct.map(r => (r._1,r._2.map(x => ItemDataArray(x-1))))

    println("Recommend 5 movie for users(only take front 20 users)")
    for ( (k, v) <- users_movie.take(20)) {
      println("=====================")
      println(k, v.mkString("|")) 
    }

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