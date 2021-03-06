package edu.cmu.ml.rtw.pra.experiments

import edu.cmu.ml.rtw.pra.graphs.PprNegativeExampleSelector
import edu.cmu.ml.rtw.users.matt.util.Dictionary
import edu.cmu.ml.rtw.users.matt.util.FakeFileUtil
import edu.cmu.ml.rtw.users.matt.util.Pair
import edu.cmu.ml.rtw.users.matt.util.TestUtil
import edu.cmu.ml.rtw.users.matt.util.TestUtil.Function

import scala.collection.mutable
import scala.collection.JavaConverters._

import java.io.BufferedReader
import java.io.StringReader

import org.scalatest._

import org.json4s._
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods.{pretty,render}

class SplitCreatorSpec extends FlatSpecLike with Matchers {

  val params: JValue =
    ("percent training" -> .3) ~
    ("relations" -> Seq("rel/1")) ~
    ("relation metadata" -> "nell") ~
    ("graph" -> "nell")
  val praBase = "/"
  val splitDir = "/splits/split_name/"
  val dataFile = "node1\tnode2\n"

  val fakeFileUtil = new FakeFileUtil
  fakeFileUtil.addFileToBeRead("/graphs/nell/num_shards.tsv", "1\n")
  fakeFileUtil.addFileToBeRead("/relation_metadata/nell/category_instances/c1", "node1\n")
  fakeFileUtil.addFileToBeRead("/relation_metadata/nell/category_instances/c2", "node2\n")
  fakeFileUtil.addFileToBeRead("/relation_metadata/nell/domains.tsv", "rel/1\tc1\n")
  fakeFileUtil.addFileToBeRead("/relation_metadata/nell/ranges.tsv", "rel/1\tc2\n")
  fakeFileUtil.addFileToBeRead("/relation_metadata/nell/relations/rel_1", "node1\tnode2\n")
  fakeFileUtil.addFileToBeRead("/graphs/nell/node_dict.tsv", "1\tnode1\n2\tnode2\n")
  fakeFileUtil.addFileToBeRead("/graphs/nell/edge_dict.tsv", "1\trel/1\n")
  fakeFileUtil.onlyAllowExpectedFiles()
  val splitCreator = new SplitCreator(params, praBase, splitDir, fakeFileUtil)

  val node_dict = new Dictionary
  node_dict.getIndex("node1")
  node_dict.getIndex("node2")

  val positiveSources = Seq(1:Integer, 1:Integer)
  val positiveTargets = Seq(1:Integer, 2:Integer)
  val negativeSources = Seq(2:Integer, 1:Integer)
  val negativeTargets = Seq(2:Integer, 2:Integer)
  // TODO(matt): need to override splitData here.
  val goodData = new Dataset(positiveSources.asJava, positiveTargets.asJava,
      negativeSources.asJava, negativeTargets.asJava) {
    override def splitData(percent: Double) = {
      println("Splitting fake data")
      val training = new Dataset(positiveSources.take(1).asJava, positiveTargets.take(1).asJava,
        negativeSources.take(1).asJava, negativeTargets.take(1).asJava)
      val testing = new Dataset(positiveSources.drop(1).asJava, positiveTargets.drop(1).asJava,
        negativeSources.drop(1).asJava, negativeTargets.drop(1).asJava)
      new Pair[Dataset, Dataset](training, testing)
    }
  }
  val badData = new FakeDatasetFactory().fromFile(null, null)

  "createNegativeExampleSelector" should "return null with no input" in {
    splitCreator.createNegativeExampleSelector(JNothing) should be(null)
  }

  it should "return a PprNegativeExampleSelector with the right input" in {
    val params: JValue = ("iterations" -> 1)
    val selector = splitCreator.createNegativeExampleSelector(params)
    selector.graphFile should be("/graphs/nell/graph_chi/edges.tsv")
    selector.numShards should be(1)
  }

  "addNegativeExampels" should "read domains and ranges correctly" in {
    val relation = "rel1"
    val domains = Map(relation -> "c1")
    val ranges = Map(relation -> "c2")
    var creator = splitCreatorWithFakeNegativeSelector(Set(1), Set(2))
    creator.addNegativeExamples(goodData, relation, domains, ranges, node_dict) should be(goodData)
    // Adding a test with the wrong sources and targets, just to be sure the test is really // working.
    creator = splitCreatorWithFakeNegativeSelector(Set(2), Set(1))
    creator.addNegativeExamples(goodData, relation, domains, ranges, node_dict) should be(badData)
  }

  it should "handle null domains and ranges" in {
    val creator = splitCreatorWithFakeNegativeSelector(null, null)
    creator.addNegativeExamples(goodData, "rel1", null, null, node_dict) should be(goodData)
  }

  it should "throw an error if the relation is missing from domain or range" in {
    val creator = splitCreatorWithFakeNegativeSelector(null, null)
    TestUtil.expectError(classOf[NoSuchElementException], new Function() {
      def call() {
        creator.addNegativeExamples(goodData, "rel1", Map(), null, node_dict) should be(goodData)
      }
    })
    TestUtil.expectError(classOf[NoSuchElementException], new Function() {
      def call() {
        creator.addNegativeExamples(goodData, "rel1", null, Map(), node_dict) should be(goodData)
      }
    })
  }

  "createSplit" should "correctly create a split" in {
    // TODO(matt): if these tests are run out of order, or another one is added after this, this
    // could easily break.  The fileUtil needs to be reset.
    fakeFileUtil.addExpectedFileWritten("/splits/split_name/in_progress", "")
    fakeFileUtil.addExpectedFileWritten("/splits/split_name/params.json", pretty(render(params)))
    fakeFileUtil.addExpectedFileWritten("/splits/split_name/relations_to_run.tsv", "rel/1\n")
    val trainingFile = "node1\tnode1\t1\nnode2\tnode2\t-1\n"
    fakeFileUtil.addExpectedFileWritten("/splits/split_name/rel_1/training.tsv", trainingFile)
    val testingFile = "node1\tnode2\t1\nnode1\tnode2\t-1\n"
    fakeFileUtil.addExpectedFileWritten("/splits/split_name/rel_1/testing.tsv", testingFile)
    var creator = splitCreatorWithFakeNegativeSelector(Set(1), Set(2))
    creator.createSplit()
    fakeFileUtil.expectFilesWritten()
  }


  def splitCreatorWithFakeNegativeSelector(expectedSources: Set[Int], expectedTargets: Set[Int]) = {
    new SplitCreator(params, praBase, splitDir, fakeFileUtil) {
      override def createNegativeExampleSelector(params: JValue) = {
        new FakeNegativeExampleSelector(expectedSources, expectedTargets)
      }
    }
  }

  class FakeNegativeExampleSelector(expectedSources: Set[Int], expectedTargets: Set[Int])
      extends PprNegativeExampleSelector(JNothing, "", 1) {
    override def selectNegativeExamples(
        data: Dataset,
        allowedSources: Set[Int],
        allowedTargets: Set[Int]): Dataset = {
      if (expectedSources == allowedSources && expectedTargets == allowedTargets) {
        goodData
      } else {
        badData
      }
    }
  }
}
