package edu.cmu.ml.rtw.pra.models

import cc.mallet.pipe.Noop
import cc.mallet.pipe.Pipe
import cc.mallet.types.Alphabet
import cc.mallet.types.FeatureVector
import cc.mallet.types.Instance
import cc.mallet.types.InstanceList

import edu.cmu.ml.rtw.pra.config.PraConfig
import edu.cmu.ml.rtw.pra.experiments.Dataset
import edu.cmu.ml.rtw.pra.features.FeatureMatrix
import edu.cmu.ml.rtw.pra.features.MatrixRow
import edu.cmu.ml.rtw.pra.features.PathType
import edu.cmu.ml.rtw.pra.features.PathTypeFactory
import edu.cmu.ml.rtw.users.matt.util.FileUtil

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * Handles learning and classification for a simple model that uses PRA
 * features.
 *
 * I thought about spending time to make this class nicer.  But then I decided that what I'm really
 * focusing on is the feature generation side of things, and the point is to use PRA features in
 * different kinds of models.  Spending too much time on making a consistent interface for just a
 * logistic regression model didn't seem to be worth it.  Maybe some day, but not now.  I've
 * thought about doing some experiments where you vary the relation extraction model (like, SVM vs.
 * LR, ranking loss instead of likelihood, different ways to handle negative evidence).  If I ever
 * get to those experiments, I'll clean up this code, but until then, I won't change what isn't
 * broken.
 */

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

  /**
   * Give a score to every row in the feature matrix, according to the given weights.
   *
   * This just applies the logistic function specified by <code>weights</code> to the feature
   * matrix, returning a score for each row in the matrix.  We convert the matrix into a map,
   * keyed by source node, to facilitate easy ranking of predictions for each source.  The lists
   * returned are not sorted yet, however.
   *
   * @param featureMatrix A feature matrix specified as a list of {@link MatrixRow} objects.
   *     Each row receives a score from the logistic function.
   * @param weights A list of feature weights, where the indices to the weights correspond to the
   *     columns of the supplied feature matrix.
   *
   * @return A map from source node to (target node, score) pairs, where the score is computed
   *     from the features in the feature matrix and the supplied weights.
   */
  def classifyInstances(featureMatrix: FeatureMatrix, weights: Seq[Double]): Map[Int, Seq[(Int, Double)]] 

  /**
   * File must be in the format "%s\t%f\n", where the string is a path description.  If the model
   * is output by outputWeights, you should be fine.
   */
  def readWeightsFromFile(filename: String, pathTypeFactory: PathTypeFactory): Seq[(PathType, Double)] 

  def classifyMatrixRow(row: MatrixRow, weights: Seq[Double]): Double 

  def matrixRowToInstance(row: MatrixRow, alphabet: Alphabet, positive: Boolean): Instance 
}