package edu.cmu.ml.rtw.pra.models

import cc.mallet.types.Alphabet
import cc.mallet.types.FeatureVector
import cc.mallet.types.Instance
import cc.mallet.types.InstanceList

import org.json4s._
import org.json4s.native.JsonMethods._

import edu.cmu.ml.rtw.pra.config.JsonHelper
import edu.cmu.ml.rtw.pra.config.PraConfig
import edu.cmu.ml.rtw.pra.experiments.Dataset
import edu.cmu.ml.rtw.pra.features.FeatureMatrix
import edu.cmu.ml.rtw.pra.features.MatrixRow

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
<<<<<<< HEAD
 * Handles learning and classification for a simple model that uses PRA
 * features.
=======
 * Handles learning and classification for models that uses PRA features.
>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c
 *
 * Note that this only deals with _feature indices_, and has no concept of path types or anything
 * else.  So you need to be sure that the feature indices don't change between training and
 * classification time, or your model will be all messed up.
 */

<<<<<<< HEAD
/**
 * ----------- abhishek ----------------------
 * Chose to implement this as an abstract class, as it allows to pass in parameters, unlike a trait.
 */
abstract class PraModel {
  /**
   * Given a feature matrix and a list of sources and targets that determines whether an
   * instances is positive or negative, train a model.
   */
  def trainModel(featureMatrix: FeatureMatrix, dataset: Dataset, featureNames: Seq[String]) 

  /** Once the model has been trained, return the parameters of the classifier
   * These are the feature weights in case of logistic regression classifier 
   * These are the alphas associated with training instances in case of svm classifier
   */
  def getParams(): Seq[Double]
  
  // TODO(matt): Clean up these three methods.  Probably the right thing to do is to put some
  // kind of class into the PraConfig object that lets you decide how to handle negative
  // evidence.
  def addNegativeEvidence(
      numPositiveExamples: Int,
      numPositiveFeatures: Int,
      negativeMatrix: FeatureMatrix,
      unseenMatrix: FeatureMatrix,
      data: InstanceList,
      alphabet: Alphabet)
      
  def sampleUnseenExamples(
      numPositiveExamples: Int,
      negativeMatrix: FeatureMatrix,
      unseenMatrix: FeatureMatrix,
      data: InstanceList,
      alphabet: Alphabet)

  def weightUnseenExamples(
      numPositiveFeatures: Int,
      negativeMatrix: FeatureMatrix,
      unseenMatrix: FeatureMatrix,
      data: InstanceList,
      alphabet: Alphabet) 
=======
abstract class PraModel(config: PraConfig, binarizeFeatures: Boolean) {
  /**
   * Given a feature matrix and a list of sources and targets that determines whether an
   * instance is positive or negative, train a model.
   */
  def train(featureMatrix: FeatureMatrix, dataset: Dataset, featureNames: Seq[String])

  // TODO(matt): this interface could probably be cleaned up a bit.
  def convertFeatureMatrixToMallet(
      featureMatrix: FeatureMatrix,
      dataset: Dataset,
      featureNames: Seq[String],
      data: InstanceList,
      alphabet: Alphabet) {
    val knownPositives = dataset.getPositiveInstances.asScala.map(x => (x.getLeft.toInt, x.getRight.toInt)).toSet
    val knownNegatives = dataset.getNegativeInstances.asScala.map(x => (x.getLeft.toInt, x.getRight.toInt)).toSet

    println("Separating into positive, negative, unseen")
    val grouped = featureMatrix.getRows().asScala.groupBy(row => {
      val sourceTarget = (row.sourceNode.toInt, row.targetNode.toInt)
      if (knownPositives.contains(sourceTarget))
        "positive"
      else if (knownNegatives.contains(sourceTarget))
        "negative"
      else
        "unseen"
    })
    val positiveMatrix = new FeatureMatrix(grouped.getOrElse("positive", Seq()).asJava)
    val negativeMatrix = new FeatureMatrix(grouped.getOrElse("negative", Seq()).asJava)
    val unseenMatrix = new FeatureMatrix(grouped.getOrElse("unseen", Seq()).asJava)
    if (config.outputMatrices && config.outputBase != null) {
      println("Outputting matrices")
      val base = config.outputBase
      config.outputter.outputFeatureMatrix(s"${base}positive_matrix.tsv", positiveMatrix, featureNames.asJava)
      config.outputter.outputFeatureMatrix(s"${base}negative_matrix.tsv", negativeMatrix, featureNames.asJava)
      config.outputter.outputFeatureMatrix(s"${base}unseen_matrix.tsv", unseenMatrix, featureNames.asJava)
    }

    println("Converting positive matrix to MALLET instances and adding to the dataset")
    // First convert the positive matrix to a scala object
    positiveMatrix.getRows().asScala
    // Then, in parallel, map the MatrixRow objects there to MALLET Instance objects
      .par.map(row => matrixRowToInstance(row, alphabet, true))
    // Then, sequentially, add them to the data object, and simultaneously count how many columns
    // there are.
      .seq.foreach(instance => {
        data.addThruPipe(instance)
      })

    println("Adding negative evidence")
    val numPositiveFeatures = positiveMatrix.getRows().asScala.map(_.columns).sum
    var numNegativeFeatures = 0
    for (negativeExample <- negativeMatrix.getRows().asScala) {
      numNegativeFeatures += negativeExample.columns
      data.addThruPipe(matrixRowToInstance(negativeExample, alphabet, false))
    }
    println("Number of positive features: " + numPositiveFeatures)
    println("Number of negative features: " + numNegativeFeatures)
    if (numNegativeFeatures < numPositiveFeatures) {
      println("Using unseen examples to make up the difference")
      val difference = numPositiveFeatures - numNegativeFeatures
      var numUnseenFeatures = 0.0
      for (unseenExample <- unseenMatrix.getRows().asScala) {
        numUnseenFeatures += unseenExample.columns
      }
      println("Number of unseen features: " + numUnseenFeatures)
      val unseenWeight = difference / numUnseenFeatures
      println("Unseen weight: " + unseenWeight)
      for (unseenExample <- unseenMatrix.getRows().asScala) {
        val unseenInstance = matrixRowToInstance(unseenExample, alphabet, false)
        data.addThruPipe(unseenInstance)
        data.setInstanceWeight(unseenInstance, unseenWeight)
      }
    }
  }
>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c

  /**
   * Give a score to every row in the feature matrix, according to the learned weights.
   *
   * @param featureMatrix A feature matrix specified as a list of {@link MatrixRow} objects.
   *     Each row receives a score from the classifier.
   *
   * @return A map from source node to (target node, score) pairs, where the score is computed
   *     from the features in the feature matrix and the learned weights.
   */
<<<<<<< HEAD
  def classifyInstances(featureMatrix: FeatureMatrix, weights: Seq[Double]): Map[Int, Seq[(Int, Double)]] 

  /**
   * File must be in the format "%s\t%f\n", where the string is a path description.  If the model
   * is output by outputWeights, you should be fine.
   */
  def readWeightsFromFile(filename: String, pathTypeFactory: PathTypeFactory): Seq[(PathType, Double)] 

  def classifyMatrixRow(row: MatrixRow, weights: Seq[Double]): Double 

  def matrixRowToInstance(row: MatrixRow, alphabet: Alphabet, positive: Boolean): Instance 
}
=======
  def classifyInstances(featureMatrix: FeatureMatrix): Map[Int, Seq[(Int, Double)]] = {
    println("Classifying instances")
    val sourceScores = new mutable.HashMap[Int, mutable.ArrayBuffer[(Int, Double)]]
    println("LR Model: size of feature matrix to classify is " + featureMatrix.size())
    for (row <- featureMatrix.getRows().asScala) {
      val score = classifyMatrixRow(row)
      sourceScores.getOrElseUpdate(row.sourceNode, new mutable.ArrayBuffer[(Int, Double)])
        .append((row.targetNode, score))
    }
    sourceScores.mapValues(_.toSeq.sortBy(x => (-x._2, x._1))).toMap
  }

  protected def classifyMatrixRow(row: MatrixRow): Double

  def matrixRowToInstance(row: MatrixRow, alphabet: Alphabet, positive: Boolean): Instance = {
    val value = if (positive) 1.0 else 0.0
    val rowValues = row.values.map(v => if (binarizeFeatures) 1 else v)
    val feature_vector = new FeatureVector(alphabet, row.pathTypes, rowValues)
    new Instance(feature_vector, value, row.sourceNode + " " + row.targetNode, null)
  }
}

object PraModelCreator {
  def create(config: PraConfig, params: JValue): PraModel = {
    val modelType = JsonHelper.extractWithDefault(params, "type", "LogisticRegressionModel")
    modelType match {
      case "logistic regression" => new LogisticRegressionModel(config, params)
      case "svm" => new SVMModel(config, params)
      case other => throw new IllegalStateException("Unrecognized model type")
    }
  }
}
>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c
