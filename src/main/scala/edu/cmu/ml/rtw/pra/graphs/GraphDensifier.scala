package edu.cmu.ml.rtw.pra.graphs

import edu.cmu.ml.rtw.pra.experiments.Driver
import edu.cmu.ml.rtw.pra.config.JsonHelper
import edu.cmu.ml.rtw.users.matt.util.Dictionary
import edu.cmu.ml.rtw.users.matt.util.FileUtil

import breeze.linalg._

import scala.collection.mutable
import scala.collection.JavaConverters._

import org.json4s._
import org.json4s.native.JsonMethods.{pretty,render,parse}

class GraphDensifier(
    praBase: String,
    graphDir: String,
    name: String,
    fileUtil: FileUtil = new FileUtil) {
  implicit val formats = DefaultFormats

  val matrixDir = s"${graphDir}${name}/"
  val paramFile = s"${matrixDir}params.json"
  val inProgressFile = s"${matrixDir}in_progress"

  val maxMatrixFileSize = 100000
  val edge_dict = new Dictionary
  edge_dict.setFromFile(graphDir + "/edge_dict.tsv")

  val node_dict = new Dictionary
  node_dict.setFromFile(graphDir + "/node_dict.tsv")

  def densifyGraph(params: JValue) {
    fileUtil.mkdirs(matrixDir)
    fileUtil.touchFile(inProgressFile)
    val param_out = fileUtil.getFileWriter(paramFile)
    param_out.write(pretty(render(params)))
    param_out.close

    val similarity_matrix_file = getSimilarityMatrixFile(params)
    println("Reading the similarity matrix")
    val similarity_matrix = readSimilarityMatrix(similarity_matrix_file)
    val test_edges: Set[(Int, Int, Int)] = (params \ "split") match {
      case JString(name) => {
        val metadata_dir =
          JsonHelper.getPathOrNameOrNull(params, "relation metadata", praBase, "relation_metadata")
        getTestEdges(graphDir, name, metadata_dir)
      }
      case JNothing => Set()
      case other => throw new IllegalStateException("split not specified correctly")
    }
    println(s"Found ${test_edges.size} test edges")
    println("Reading the graph")
    val edge_vectors = readGraphEdges(graphDir + "/graph_chi/edges.tsv", test_edges)
    println(s"Creating (${edge_vectors.size}) dense entity pair vectors")
    val dense_edge_vectors = edge_vectors.par.map(x => (x._1, similarity_matrix * x._2))
    println(s"Rekey-ing by relation")
    val relation_matrices = dense_edge_vectors.flatMap(x => {
      val entries = new mutable.ArrayBuffer[(Int, Int, Int, Double)]
      var offset = 0
      while (offset < x._2.activeSize) {
        val relation = x._2.indexAt(offset)
        val value = x._2.valueAt(offset)
        entries += Tuple4(relation, x._1._1, x._1._2, value)
        offset += 1
      }
      entries.toSeq
    }).groupBy(_._1).mapValues(x => x.map(y => (y._2, y._3, y._4)).seq.toSet).seq
    println("Outputting relation matrices")
    val edges_to_write = new mutable.ArrayBuffer[Set[(Int, Int, Double)]]
    var start_relation = 1
    var edges_so_far = 0
    for (i <- 1 until edge_dict.getNextIndex) {
      val matrix = relation_matrices.getOrElse(i, Set())
      if (edges_so_far > 0 && edges_so_far + matrix.size > maxMatrixFileSize) {
        writeEdgesSoFar(start_relation, i - 1, edges_to_write.toSeq)
        edges_to_write.clear()
        start_relation = i
        edges_so_far = 0
      }
      edges_to_write += matrix
      edges_so_far += matrix.size
    }
    if (edges_to_write.size > 0) {
      writeEdgesSoFar(start_relation, edge_dict.getNextIndex, edges_to_write)
    }
    println("Done creating matrices")
    fileUtil.deleteFile(inProgressFile)
  }

  def emptyMatrixBuilder(): CSCMatrix.Builder[Double] = {
    new CSCMatrix.Builder[Double](node_dict.getNextIndex, node_dict.getNextIndex)
  }

  def writeEdgesSoFar(_start_relation: Int, end_relation: Int, edges_to_write: Seq[Set[(Int, Int, Double)]]) {
    var start_relation = _start_relation
    var filename = matrixDir + start_relation
    if (end_relation > start_relation) {
      filename += "-" + end_relation
    }
    val writer = fileUtil.getFileWriter(filename)
    for (matrix <- edges_to_write) {
      writer.write("Relation " + start_relation + "\n")
      for (entry <- matrix) {
        writer.write(entry._1 + "\t" + entry._2 + "\t" + entry._3 + "\n")
      }
      start_relation += 1
    }
    writer.close()
  }

  def getTestEdges(graph_dir: String, split_name: String, metadata: String): Set[(Int, Int, Int)] = {
    val node_dict = new Dictionary
    node_dict.setFromFile(graph_dir + "/node_dict.tsv")
    val edge_dict = new Dictionary
    edge_dict.setFromFile(graph_dir + "/edge_dict.tsv")
    println(s"Metadata directory: $metadata")
    val inverses = Driver.createInverses(metadata, edge_dict, fileUtil)
    println(s"Inverses size: ${inverses.size}")
    // TODO(matt): don't I have some common code for reading a split?
    val split_dir = s"${praBase}splits/${split_name}/"
    val relations = fileUtil.readLinesFromFile(s"${split_dir}relations_to_run.tsv").asScala
    relations.flatMap(relation => {
      val rel_index = edge_dict.getIndex(relation)
      val has_inverse = inverses.contains(rel_index)
      val inverse_index = if (has_inverse) inverses(rel_index) else -1
      val test_file = s"${split_dir}${relation}/testing.tsv"
      val instances = fileUtil.readLinesFromFile(test_file).asScala
      instances.flatMap(instance => {
        val fields = instance.split("\t")
        val source_index = node_dict.getIndex(fields(0))
        val target_index = node_dict.getIndex(fields(1))
        if (has_inverse) {
          List(Tuple3(source_index, target_index, rel_index),
               Tuple3(target_index, source_index, inverse_index))
        } else {
          List(Tuple3(source_index, target_index, rel_index))
        }
      })
    }).toSet
  }

  def readGraphEdges(edge_file: String, test_edges: Set[(Int, Int, Int)]): Map[(Int, Int), SparseVector[Double]] = {
    val edges = new mutable.HashMap[(Int, Int), Set[Int]]().withDefaultValue(Set())
    var seen_test_edges = 0
    for (line <- fileUtil.readLinesFromFile(edge_file).asScala) {
      val fields = line.split("\t")
      val source = fields(0).toInt
      val target = fields(1).toInt
      val relation = fields(2).toInt
      if (!test_edges.contains((source, target, relation))) {
        edges.update((source, target), edges(source, target) + relation)
      } else {
        seen_test_edges += 1
      }
    }
    println(s"Saw $seen_test_edges test edges")
    edges.map(x => (x._1, createSparseVector(x._2))).toMap
  }

  def createSparseVector(relations: Set[Int]): SparseVector[Double] = {
    val builder = new VectorBuilder[Double](edge_dict.getNextIndex)
    for (relation <- relations) {
      builder.add(relation, 1.0)
    }
    builder.toSparseVector
  }

  def readSimilarityMatrix(filename: String): CSCMatrix[Double] = {
    val entries = new mutable.ListBuffer[(Int, Int, Double)]
    for (line <- fileUtil.readLinesFromFile(filename).asScala) {
      val fields = line.split("\t")
      val relation1 = edge_dict.getIndex(fields(0))
      val relation2 = edge_dict.getIndex(fields(1))
      val weight = fields(2).toDouble
      entries += Tuple3(relation1, relation2, weight)
    }
    val builder = new CSCMatrix.Builder[Double](rows=edge_dict.getNextIndex, cols=edge_dict.getNextIndex)
    for (entry <- entries) {
      builder.add(entry._1, entry._2, entry._3)
    }
    for (i <- 0 until edge_dict.getNextIndex) {
      builder.add(i, i, 1.0)
    }
    builder.result
  }

  def getSimilarityMatrixFile(params: JValue): String = {
    (params \ "similarity matrix") match {
      case JString(path) if (path.startsWith("/")) => path
      case jobj: JObject => {
        val name = (jobj \ "name").extract[String]
        val embeddingsName = getNameFromEmbeddings(jobj \ "embeddings")
        s"${praBase}embeddings/${embeddingsName}/${name}/matrix.tsv"
      }
    }
  }

  def getNameFromEmbeddings(params: JValue): String = {
    params match {
      case JString(name) => name
      case jval => (params \ "name").extract[String]
    }
  }
}
