package edu.cmu.ml.rtw.pra.graphs

import edu.cmu.ml.rtw.pra.config.JsonHelper
import edu.cmu.ml.rtw.users.matt.util.Dictionary
import edu.cmu.ml.rtw.users.matt.util.FileUtil
import edu.cmu.ml.rtw.users.matt.util.IntTriple
import edu.cmu.ml.rtw.users.matt.util.Pair

import java.io.BufferedReader
import java.io.FileWriter

import scala.collection.JavaConverters._
import scala.collection.mutable

import org.json4s._

class RelationSet(params: JValue, fileUtil: FileUtil = new FileUtil) {
  implicit val formats = DefaultFormats

  // Fields dealing with the relations themselves.

  // The file containing the relation triples.
  val relationFile = (params \ "relation file").extract[String]

  // If not null, prepend this prefix to all relation strings in this set.
  val relationPrefix = JsonHelper.extractWithDefault(params, "relation prefix", null: String)

  // KB relations or surface relations?  The difference is only in the alias relation format (and
  // in the relation file format).
  val isKb = JsonHelper.extractWithDefault(params, "is kb", false)

  // Fields specific to KB relations.

  // If this is a KB relation set, this file contains the mapping from entities to noun phrases.
  val aliasFile = JsonHelper.extractWithDefault(params, "alias file", null: String)

  // What should we call the alias relation?  Defaults to "@ALIAS@".  Note that we specify this for
  // each set of _KB_ relations, not for each set of _surface_ relations.  If you want something
  // more complicated, like a different name for each (KB, surface) relation set pair, you'll have
  // to approximate it with relation prefixes.
  val aliasRelation = JsonHelper.extractWithDefault(params, "alias relation", "@ALIAS@")

  // Determines how we try to read the alias file.  We currently allow two values: "freebase" and
  // "nell".  The freebase format is a little complicated the nell format is just a list of
  // (concept, noun phrase) pairs that we read into a map.  The "nell" format is generic enough
  // that it could be used for more than just NELL KBs, probably.
  val aliasFileFormat = JsonHelper.extractWithDefault(params, "alias file format", null: String)


  // Fields specific to surface relations.

  // If this is true, we only add the alias edges that get generated from the set, not the
  // relations themselves (either from a KB relation set or a surface relation set).  This is still
  // somewhat experimental, but there are some cases where you might want to try this.
  val aliasesOnly = JsonHelper.extractWithDefault(params, "aliases only", false)

  // Fields for embeddings.

  // The file where we can find embeddings for the relations in this set.
  val embeddingsFile = JsonHelper.extractWithDefault(params, "embeddings file", null: String)

  // If we are embedding the edges, should we replace the original edges or just augment them?
  val keepOriginalEdges = JsonHelper.extractWithDefault(params, "keep original edges", false)

  val replaceRelationsWith = JsonHelper.extractWithDefault(params, "replace relations with", null: String)

  /**
   * Get the set of aliases specified by this KB relation set.
   */
  def getAliases(): Map[String, List[String]] = {
    if (aliasFile == null) {
      Map()
    } else {
      getAliasesFromReader(fileUtil.getBufferedReader(aliasFile))
    }
  }

  def getAliasesFromReader(reader: BufferedReader): Map[String, List[String]] = {
    aliasFileFormat match {
      case "freebase" => {
        fileUtil.readMapListFromTsvReader(reader, 3, false, new FileUtil.LineFilter() {
          override def filter(fields: Array[String]): Boolean = {
            if (fields.length != 4) return true
            return false
          }
        }).asScala.mapValues(_.asScala.toList).toMap
      }
      case "nell" => {
        fileUtil.readInvertedMapListFromTsvReader(reader, 1000000)
          .asScala.mapValues(_.asScala.toList).toMap
      }
      case other => throw new IllegalStateException("Unrecognized alias file format")
    }
  }

  def writeRelationEdgesToGraphFile(
      intEdgeFile: FileWriter,
      seenTriples: mutable.HashSet[(Int, Int, Int)],
      prefixOverride: String,
      seenNps: mutable.HashSet[(String)],
      aliases: Seq[(String, Map[String, List[String]])],
      nodeDict: Dictionary,
      edgeDict: Dictionary): Int = {
    writeRelationEdgesFromReader(
      fileUtil.getBufferedReader(relationFile),
      loadEmbeddings(),
      seenTriples,
      prefixOverride,
      seenNps,
      aliases,
      intEdgeFile,
      nodeDict,
      edgeDict)
  }

  def writeRelationEdgesFromReader(
      reader: BufferedReader,
      embeddings: Map[String, List[String]],
      seenTriples: mutable.HashSet[(Int, Int, Int)],
      prefixOverride: String,
      seenNps: mutable.HashSet[(String)],
      aliases: Seq[(String, Map[String, List[String]])],
      writer: FileWriter,
      nodeDict: Dictionary,
      edgeDict: Dictionary): Int = {
    println(s"Adding edges from relation file $relationFile")
    val prefix = {
      if (prefixOverride != null) {
        prefixOverride
      } else if (relationPrefix != null) {
        relationPrefix
      } else {
        ""
      }
    }
    var line: String = null
    var i = 0
    var numEdges = 0
    while ({ line = reader.readLine(); line != null }) {
      i += 1
      fileUtil.logEvery(1000000, i)
      val fields = line.split("\t")
      var relation: String = null
      var arg1: String = null
      var arg2: String = null
      // TODO(matt): Maybe make this relation file format a configurable field?
      if (isKb) {
        // KB relations are formated (S, O, V).
        arg1 = fields(0)
        arg2 = fields(1)
        relation = fields(2)
      } else {
        // And surface relations are formatted (S, V, O).
        arg1 = fields(0)
        relation = fields(1)
        arg2 = fields(2)
      }
      val ind1 = nodeDict.getIndex(arg1)
      val ind2 = nodeDict.getIndex(arg2)

      if (!isKb) {
        numEdges += addAliasEdges(arg1, ind1, seenNps, writer, nodeDict, edgeDict, aliases)
        numEdges += addAliasEdges(arg2, ind2, seenNps, writer, nodeDict, edgeDict, aliases)
      }

      if (!aliasesOnly) {
        val replaced = replaceRelation(relation)
        val relationEdges = getEmbeddedRelations(replaced, embeddings)
        for (relation <- relationEdges) {
          val prefixed_relation = prefix + relation
          val relation_index = edgeDict.getIndex(prefixed_relation)
          writeEdgeIfUnseen(ind1, ind2, relation_index, seenTriples, writer)
          numEdges += 1
        }
      }
    }
    numEdges
  }

  def writeEdgeIfUnseen(
      arg1: Int,
      arg2: Int,
      rel: Int,
      seenTriples: mutable.HashSet[(Int, Int, Int)],
      writer: FileWriter) {
    if (seenTriples != null) {
      val triple = (arg1, arg2, rel)
      if (seenTriples.contains(triple)) return
      seenTriples.add(triple)
    }
    writer.write(s"${arg1}\t${arg2}\t${rel}\n")
  }

  def addAliasEdges(
      np: String,
      np_index: Int,
      seenNps: mutable.HashSet[String],
      writer: FileWriter,
      nodeDict: Dictionary,
      edgeDict: Dictionary,
      aliases: Seq[(String, Map[String, List[String]])]): Int = {
    var numEdges = 0
    if (seenNps.contains(np)) return numEdges
    seenNps.add(np)
    for (aliasSet <- aliases) {
      val aliasRelation = aliasSet._1
      val aliasIndex = edgeDict.getIndex(aliasRelation)
      val currentAliases = aliasSet._2
      val concepts = currentAliases.getOrElse(np, Nil)
      for (concept <- concepts) {
        val concept_index = nodeDict.getIndex(concept)
        writer.write(s"${np_index}\t${concept_index}\t${aliasIndex}\n")
        numEdges += 1
      }
    }
    return numEdges
  }

  def replaceRelation(relation: String): String = {
    if (replaceRelationsWith != null) {
      replaceRelationsWith
    } else {
      relation
    }
  }

  def getEmbeddedRelations(relation: String, embeddings: Map[String, List[String]]) = {
    if (embeddings != null) {
      if (keepOriginalEdges) {
        relation :: embeddings.getOrElse(relation, Nil)
      } else {
        embeddings.getOrElse(relation, List(relation))
      }
    } else {
      List(relation)
    }
  }

  def loadEmbeddings(): Map[String, List[String]] = {
    if (embeddingsFile != null) {
      System.out.println("Reading embeddings from file " + embeddingsFile)
      fileUtil.readMapListFromTsvFile(embeddingsFile).asScala.mapValues(_.asScala.toList).toMap
    } else {
      null
    }
  }
}
