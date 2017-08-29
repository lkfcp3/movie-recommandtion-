name := "Simple Project"

version := "1.0"

scalaVersion := "2.11.8"

unmanagedJars in Compile += file("~/spark/")

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.1.0"
libraryDependencies += "org.jblas" % "jblas" % "1.2.4"
