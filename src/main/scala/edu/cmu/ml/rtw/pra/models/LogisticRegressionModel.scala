package edu.cmu.ml.rtw.pra.models

import cc.mallet.pipe.Noop
<<<<<<< HEAD
import cc.mallet.pipe.Pipe
import cc.mallet.types.Alphabet
import cc.mallet.types.FeatureVector
import cc.mallet.types.Instance
import cc.mallet.types.InstanceList

=======
import cc.mallet.types.Alphabet
import cc.mallet.types.InstanceList

import org.json4s._
import org.json4s.native.JsonMethods._

import edu.cmu.ml.rtw.pra.config.JsonHelper
>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c
import edu.cmu.ml.rtw.pra.config.PraConfig
import edu.cmu.ml.rtw.pra.experiments.Dataset
import edu.cmu.ml.rtw.pra.features.FeatureMatrix
import edu.cmu.ml.rtw.pra.features.MatrixRow
<<<<<<< HEAD
import edu.cmu.ml.rtw.pra.features.PathType
import edu.cmu.ml.rtw.pra.features.PathTypeFactory
import edu.cmu.ml.rtw.users.matt.util.FileUtil
=======
>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c

import scala.collection.JavaConverters._
import scala.collection.mutable

<<<<<<< HEAD
class LogisticRegressionModel(config: PraConfig, l1Weight: Double, l2Weight: Double, binarizeFeatures: Boolean)
      extends PraModel{
  
  // initializes to an empty sequence
  var lrWeights: Seq[Double] = Seq()
  
=======
class LogisticRegressionModel(config: PraConfig, params: JValue)
    extends PraModel(config, JsonHelper.extractWithDefault(params, "binarize features", false)) {
  val allowedParams = Seq("type", "l1 weight", "l2 weight", "binarize features")
  JsonHelper.ensureNoExtras(params, "pra parameters -> learning", allowedParams)

  val l1Weight = JsonHelper.extractWithDefault(params, "l1 weight", 0.0)

  val l2Weight = JsonHelper.extractWithDefault(params, "l2 weight", 0.0)

  // initializes to an empty sequence
  var lrWeights: Seq[Double] = Seq()

>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c
  /**
   * Given a feature matrix and a list of sources and targets that determines whether an
   * instances is positive or negative, train a logistic regression classifier.
   */
<<<<<<< HEAD
  def trainModel(featureMatrix: FeatureMatrix, dataset: Dataset, featureNames: Seq[String]) = {
    println("Learning feature weights")
    println("Prepping training data")
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
=======
  override def train(featureMatrix: FeatureMatrix, dataset: Dataset, featureNames: Seq[String]) = {
    println("Learning feature weights")
    println("Prepping training data")
>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c

    println("Creating alphabet")
    // Set up some mallet boiler plate so we can use Burr's ShellClassifier
    val pipe = new Noop()
    val data = new InstanceList(pipe)
    val alphabet = new Alphabet(featureNames.asJava.toArray())

<<<<<<< HEAD
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
    addNegativeEvidence(positiveMatrix.size,
                        positiveMatrix.getRows().asScala.map(_.columns).sum,
                        negativeMatrix,
                        unseenMatrix,
                        data,
                        alphabet)
=======
    convertFeatureMatrixToMallet(featureMatrix, dataset, featureNames, data, alphabet)

>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c
    println("Creating the MalletLogisticRegression object")
    val lr = new MalletLogisticRegression(alphabet)
    if (l2Weight != 0.0) {
      println("Setting L2 weight to " + l2Weight)
      lr.setL2wt(l2Weight)
    }
    if (l1Weight != 0.0) {
      println("Setting L1 weight to " + l1Weight)
      lr.setL1wt(l1Weight)
    }

    // Finally, we train.  All that prep and everything that follows is really just to get
    // ready for and pass on the output of this one line.
    println("Training the classifier")
    lr.train(data)
    val features = lr.getSparseFeatures()
    val params = lr.getSparseParams()
    val bias = lr.getBias()
    val weights = new mutable.ArrayBuffer[Double]()
    var j = 0
    for (i <- 0 until featureNames.size) {
      if (j >= features.length) {
        weights += 0.0
      } else if (features(j) > i) {
        weights += 0.0
      } else if (features(j) == i) {
        weights += params(j)
        j += 1
      }
    }
    println("Outputting feature weights")
    if (config.outputBase != null) {
      val javaWeights = weights.map(java.lang.Double.valueOf).asJava
      config.outputter.outputWeights(config.outputBase + "weights.tsv", javaWeights, featureNames.asJava)
    }
    lrWeights = weights.toSeq
  }
<<<<<<< HEAD
  
  def getParams(): Seq[Double] = lrWeights

  // TODO(matt): Clean up these three methods.  Probably the right thing to do is to put some
  // kind of class into the PraConfig object that lets you decide how to handle negative
  // evidence.
  def addNegativeEvidence(
      numPositiveExamples: Int,
      numPositiveFeatures: Int,
      negativeMatrix: FeatureMatrix,
      unseenMatrix: FeatureMatrix,
      data: InstanceList,
      alphabet: Alphabet) {
    // sampleUnseenExamples(numPositiveExamples, negativeMatrix, unseenMatrix, data, alphabet)
    weightUnseenExamples(numPositiveFeatures, negativeMatrix, unseenMatrix, data, alphabet)
  }

  def sampleUnseenExamples(
      numPositiveExamples: Int,
      negativeMatrix: FeatureMatrix,
      unseenMatrix: FeatureMatrix,
      data: InstanceList,
      alphabet: Alphabet) {
    unseenMatrix.shuffle()
    for (i <- 0 until numPositiveExamples) {
      data.addThruPipe(matrixRowToInstance(unseenMatrix.getRow(i), alphabet, false))
    }
  }

  def weightUnseenExamples(
      numPositiveFeatures: Int,
      negativeMatrix: FeatureMatrix,
      unseenMatrix: FeatureMatrix,
      data: InstanceList,
      alphabet: Alphabet) {
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

  /**
   * Give a score to every row in the feature matrix, according to the given weights.
   *
   * This just applies the logistic function specified by <code>weights</code> to the feature
   * matrix, returning a score for each row in the matrix.  We convert the matrix into a map,
   * keyed by source node, to facilitate easy ranking of predictions for each source.  The lists
   * returned are not sorted yet, however (looks to be sorted ?).
   *
   * @param featureMatrix A feature matrix specified as a list of {@link MatrixRow} objects.
   *     Each row receives a score from the logistic function.
   * @param weights A list of feature weights, where the indices to the weights correspond to the
   *     columns of the supplied feature matrix.
   *
   * @return A map from source node to (target node, score) pairs, where the score is computed
   *     from the features in the feature matrix and the supplied weights.
   */
  def classifyInstances(featureMatrix: FeatureMatrix, weights: Seq[Double]): Map[Int, Seq[(Int, Double)]] = {
    println("Classifying instances")
    val sourceScores = new mutable.HashMap[Int, mutable.ArrayBuffer[(Int, Double)]]
    println("LR Model: size of feature matrix to classify is " + featureMatrix.size())
    for (row <- featureMatrix.getRows().asScala) {
      val score = classifyMatrixRow(row, weights)
      sourceScores.getOrElseUpdate(row.sourceNode, new mutable.ArrayBuffer[(Int, Double)])
        .append((row.targetNode, score))
    }
    sourceScores.mapValues(_.toSeq.sortBy(x => (-x._2, x._1))).toMap
  }

  /**
   * File must be in the format "%s\t%f\n", where the string is a path description.  If the model
   * is output by outputWeights, you should be fine.
   */
  def readWeightsFromFile(filename: String, pathTypeFactory: PathTypeFactory): Seq[(PathType, Double)] = {
    val lines = new FileUtil().readLinesFromFile(filename).asScala
    lines.map(line => {
      val fields = line.split("\t")
      val description = fields(0)
      val weight = fields(1).toDouble
      val pathType = pathTypeFactory.fromString(description)
      (pathType, weight)
    })
  }

  def classifyMatrixRow(row: MatrixRow, weights: Seq[Double]) = {
    val features = row.values.zip(row.pathTypes)
    features.map(f => {
      if (f._2 < weights.size)
        f._1 * weights(f._2)
=======

  override def classifyMatrixRow(row: MatrixRow) = {
    val features = row.values.zip(row.pathTypes)
    features.map(f => {
      if (f._2 < lrWeights.size)
        f._1 * lrWeights(f._2)
>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c
      else
        0.0
    }).sum
  }
<<<<<<< HEAD

  def matrixRowToInstance(row: MatrixRow, alphabet: Alphabet, positive: Boolean): Instance = {
    val value = if (positive) 1.0 else 0.0
    val rowValues = row.values.map(v => if (binarizeFeatures) 1 else v)
    val feature_vector = new FeatureVector(alphabet, row.pathTypes, rowValues)
    new Instance(feature_vector, value, row.sourceNode + " " + row.targetNode, null)
  }

}
=======
}
>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c
