package edu.cmu.ml.rtw.pra.experiments

import edu.cmu.ml.rtw.users.matt.util.Dictionary
import edu.cmu.ml.rtw.users.matt.util.FileUtil
import edu.cmu.ml.rtw.pra.config.JsonHelper
import edu.cmu.ml.rtw.pra.config.PraConfig
import edu.cmu.ml.rtw.pra.config.SpecFileReader
import edu.cmu.ml.rtw.pra.features.PraFeatureGenerator
import edu.cmu.ml.rtw.pra.features.SubgraphFeatureGenerator
import edu.cmu.ml.rtw.pra.graphs.GraphCreator
import edu.cmu.ml.rtw.pra.graphs.GraphDensifier
import edu.cmu.ml.rtw.pra.graphs.GraphExplorer
import edu.cmu.ml.rtw.pra.graphs.PcaDecomposer
import edu.cmu.ml.rtw.pra.graphs.SimilarityMatrixCreator
<<<<<<< HEAD
import edu.cmu.ml.rtw.pra.models.LogisticRegressionModel
import edu.cmu.ml.rtw.pra.models.SVMModel
=======
import edu.cmu.ml.rtw.pra.models.PraModelCreator
>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c
import edu.cmu.ml.rtw.users.matt.util.Pair

import scala.collection.JavaConverters._
import scala.collection.mutable

import org.json4s._
import org.json4s.native.JsonMethods.{pretty,render,parse}

// TODO(matt): This class is a mess.  It needs some major refactoring, splitting this into several
// parts, and tests for each of those parts.
class Driver(praBase: String, fileUtil: FileUtil = new FileUtil()) {
  implicit val formats = DefaultFormats

  def runPra(_outputBase: String, params: JValue) {
    // The "create" key is special - it's not used for anything here, but if there's some object
    // you want to create with a PRA mode of "no op", and can't or don't want to put the object in
    // the proper nested place, you can put it under "create", and it will be found by the
    // "filterField" calls below.  This will work for creating embeddings, similarity matrices, and
    // (maybe) denser matrices.
    val baseKeys = Seq("graph", "split", "relation metadata", "pra parameters", "create")
    JsonHelper.ensureNoExtras(params, "base", baseKeys)
    val outputBase = fileUtil.addDirectorySeparatorIfNecessary(_outputBase)
    fileUtil.mkdirOrDie(outputBase)

    // We create the graph first here, because we allow a "no op" PRA mode, which means just create
    // the graph and quit.  But we have to do this _after_ we create the output directory, or we
    // could get two threads trying to do the same experiment when one of them has to create a
    // graph first.  We'll delete the output directory in the case of a no op.
    createGraphIfNecessary(params \ "graph")

    // And these are all part of "creating the graph", basically, they just deal with augmenting
    // the graph by doing some factorization.
    createEmbeddingsIfNecessary(params)
    createSimilarityMatricesIfNecessary(params)
    createDenserMatricesIfNecessary(params)

    createSplitIfNecessary(params \ "split")

    val mode = JsonHelper.extractWithDefault(params \ "pra parameters", "mode", "learn models")
    if (mode == "no op") {
      fileUtil.deleteFile(outputBase)
      return
    }

    val metadataDirectory: String = (params \ "relation metadata") match {
      case JNothing => null
      case JString(path) if (path.startsWith("/")) => fileUtil.addDirectorySeparatorIfNecessary(path)
      case JString(name) => s"${praBase}relation_metadata/${name}/"
      case other => throw new IllegalStateException("relation metadata parameter must be either "
        + "a string or absent")
    }
    val splitsDirectory = (params \ "split") match {
      case JString(path) if (path.startsWith("/")) => fileUtil.addDirectorySeparatorIfNecessary(path)
      case JString(name) => s"${praBase}splits/${name}/"
      case jval => s"${praBase}splits/" + (jval \ "name").extract[String] + "/"
    }

    val start_time = System.currentTimeMillis

    val baseBuilder = new PraConfig.Builder()
    var writer = fileUtil.getFileWriter(outputBase + "settings.txt")
    writer.write("Parameters used:\n")
    writer.write(pretty(render(params)))
    writer.write("\n")
    writer.close()

    // This takes care of setting everything in the config builder that is consistent across
    // relations.
    Driver.initializeGraphParameters(getGraphDirectory(params), baseBuilder)

    var nodeNames: java.util.Map[String, String] = null
    if (metadataDirectory != null && fileUtil.fileExists(metadataDirectory + "node_names.tsv")) {
      nodeNames = fileUtil.readMapFromTsvFile(metadataDirectory + "node_names.tsv", true)
    }
    baseBuilder.setOutputter(new Outputter(baseBuilder.nodeDict, baseBuilder.edgeDict, nodeNames))

    val baseConfig = baseBuilder.noChecks().build()

    val relationsFile = splitsDirectory + "relations_to_run.tsv"
    for (relation <- fileUtil.readLinesFromFile(relationsFile).asScala) {
      val relation_start = System.currentTimeMillis
      val builder = new PraConfig.Builder(baseConfig)
      builder.setRelation(relation)
      println("\n\n\n\nRunning PRA for relation " + relation)
      Driver.parseRelationMetadata(metadataDirectory, relation, mode, builder, outputBase)

      val outdir = fileUtil.addDirectorySeparatorIfNecessary(outputBase + relation)
      fileUtil.mkdirs(outdir)
      builder.setOutputBase(outdir)

      if (mode == "learn models") {
        learnModels(params, splitsDirectory, metadataDirectory, relation, builder)
      } else if (mode == "explore graph") {
        exploreGraph(params, builder.noChecks().build(), splitsDirectory)
      } else {
        throw new IllegalStateException("Unrecognized (or unspecified) mode!")
      }
      val relation_end = System.currentTimeMillis
      val millis = relation_end - relation_start
      var seconds = (millis / 1000).toInt
      val minutes = seconds / 60
      seconds = seconds - minutes * 60
      writer = fileUtil.getFileWriter(outputBase + "settings.txt", true)  // true -> append to the file.
      writer.write(s"Time for relation $relation: $minutes minutes and $seconds seconds\n")
      writer.close()
    }
    val end_time = System.currentTimeMillis
    val millis = end_time - start_time
    var seconds = (millis / 1000).toInt
    val minutes = seconds / 60
    seconds = seconds - minutes * 60
    writer = fileUtil.getFileWriter(outputBase + "settings.txt", true)  // true -> append to the file.
    writer.write("PRA appears to have finished all relations successfully\n")
    writer.write(s"Total time: $minutes minutes and $seconds seconds\n")
    writer.close()
    System.out.println(s"Took $minutes minutes and $seconds seconds")
  }

  // TODO(matt): as part of the refactoring mentioned at the top of this file, maybe these methods
  // should be moved to be members of the base trait object (e.g.,
  // FeatureGenerator.create(params)), instead of a member of Driver or whatever other calling
  // class needs to create the object.
  def createFeatureGenerator(praParams: JValue, config: PraConfig) = {
    val featureType = JsonHelper.extractWithDefault(praParams \ "features", "type", "pra")
    println("feature type being used is " + featureType)
    featureType match {
      case "pra" => new PraFeatureGenerator(praParams \ "features", praBase, config, fileUtil)
      case "subgraphs" => new SubgraphFeatureGenerator(praParams \ "features", praBase, config, fileUtil)
      case other => throw new IllegalStateException("Illegal feature type!")
    }
    
  }

  def learnModels(
      params: JValue,
      splitsDirectory: String,
      metadataDirectory: String,
      relation: String,
      builder: PraConfig.Builder) {
    val doCrossValidation = Driver.initializeSplit(
      splitsDirectory,
      metadataDirectory,
      relation,
      builder,
      new DatasetFactory(),
      fileUtil)
    val praParams = params \ "pra parameters"
    val praParamKeys = Seq("mode", "features", "learning")
    JsonHelper.ensureNoExtras(praParams, "pra parameters", praParamKeys)

    // Split the data if we're doing cross validation instead of a fixed split.
    if (doCrossValidation) {
      val config = builder.build()
      val splitData = config.allData.splitData(config.percentTraining)
      val trainingData = splitData.getLeft()
      val testingData = splitData.getRight()
      config.outputter.outputSplitFiles(config.outputBase, trainingData, testingData)
      builder.setAllData(null)
      builder.setPercentTraining(0)
      builder.setTrainingData(trainingData)
      builder.setTestingData(testingData)
    }

    val config = builder.build()

    // Now we actually run PRA.

    // First we get features.
    val generator = createFeatureGenerator(praParams, config)
    val trainingMatrix = generator.createTrainingMatrix(config.trainingData)

    // Then we train a model.  It'd be nice here to have all of this parameter stuff pushed
    // down into the PraModel, but PraModel is currently a java class, which doesn't play
    // nicely with json4s.
<<<<<<< HEAD
    
    // TODO: PraModel is a scala abstract class now. Check what changes
    val learningParams = praParams \ "learning"
    val learningParamKeys = Seq("l1 weight", "l2 weight", "binarize features")
    JsonHelper.ensureNoExtras(learningParams, "pra parameters -> learning", learningParamKeys)
    val l1Weight = JsonHelper.extractWithDefault(learningParams, "l1 weight", 1.0)
    val l2Weight = JsonHelper.extractWithDefault(learningParams, "l2 weight", 0.05)
    val binarize = JsonHelper.extractWithDefault(learningParams, "binarize features", false)
    //val model = new LogisticRegressionModel(config, l1Weight, l2Weight, binarize)
    val model = new SVMModel(config, l1Weight, l2Weight, binarize)
    val featureNames = generator.getFeatureNames()
    model.trainModel(trainingMatrix, config.trainingData, featureNames)
    val weights = model.getParams()
    /* Don't remove zeroWeight features when using an SVM model */
    //val finalWeights = generator.removeZeroWeightFeatures(weights)
    val finalWeights = weights
=======

    val learningParams = praParams \ "learning"
    val model = PraModelCreator.create(config, learningParams)
    val featureNames = generator.getFeatureNames()
    model.train(trainingMatrix, config.trainingData, featureNames)
>>>>>>> b33637150891edc577e99c8a762ffb8de48ac39c

    // Then we test the model.
    // TODO(matt): if we don't care about removing zero weight features anymore (and it's probably
    // not worth it, anyway), we could feasibly just generate the training and test matrices at the
    // same time, and because of how GraphChi works, that would save us considerable time.
    val testMatrix = generator.createTestMatrix(config.testingData)
    val scores = model.classifyInstances(testMatrix)
    val javaScores = scores.mapValues(_.map(x => {
      Pair.makePair(Integer.valueOf(x._1), java.lang.Double.valueOf(x._2))
    }).asJava).map(x => (Integer.valueOf(x._1), x._2)).asJava
    config.outputter.outputScores(config.outputBase + "scores.tsv", javaScores, config)
  }

  def exploreGraph(params: JValue, config: PraConfig, splitsDirectory: String) {
    val praParams = params \ "pra parameters"
    val praParamKeys = Seq("mode", "explore", "data")
    JsonHelper.ensureNoExtras(praParams, "pra parameters", praParamKeys)

    val dataToUse = JsonHelper.extractWithDefault(praParams, "data", "both")
    val datasetFactory = new DatasetFactory()
    val fixed = config.relation.replace("/", "_")
    val data = if (dataToUse == "both") {
      val trainingFile = s"${splitsDirectory}${fixed}/training.tsv"
      val trainingData = if (fileUtil.fileExists(trainingFile))
        datasetFactory.fromFile(trainingFile, config.nodeDict) else null
      val testingFile = s"${splitsDirectory}${fixed}/testing.tsv"
      val testingData = if (fileUtil.fileExists(testingFile))
        datasetFactory.fromFile(testingFile, config.nodeDict) else null
      if (trainingData == null && testingData == null) {
        throw new IllegalStateException("Neither training file nor testing file exists for " +
          "relation " + config.relation)
      }
      if (trainingData == null) {
        testingData
      } else if (testingData == null) {
        trainingData
      } else {
        trainingData.merge(testingData)
      }
    } else {
      val inputFile = s"${splitsDirectory}${fixed}/${dataToUse}.tsv"
      datasetFactory.fromFile(inputFile, config.nodeDict)
    }

    val explorer = new GraphExplorer(praParams \ "explore", config)
    val pathCountMap = explorer.findConnectingPaths(data)
    val javaMap = pathCountMap.map(entry => {
      val key = new Pair[Integer, Integer](entry._1._1, entry._1._2)
      val value = entry._2.mapValues(x => Integer.valueOf(x)).asJava
      (key, value)
    }).asJava
    config.outputter.outputPathCountMap(config.outputBase, "path_count_map.tsv", javaMap, data)
  }

  def createGraphIfNecessary(params: JValue) {
    var graph_name = ""
    var params_specified = false
    // First, is this just a path, or do the params specify a graph name?  If it's a path, we'll
    // just use the path as is.  Otherwise, we have some processing to do.
    params match {
      case JString(path) if (path.startsWith("/")) => {
        if (!fileUtil.fileExists(path)) {
          throw new IllegalStateException("Specified path to graph does not exist!")
        }
      }
      case JString(name) => graph_name = name
      case jval => {
        graph_name = (jval \ "name").extract[String]
        params_specified = true
      }
    }
    if (graph_name != "") {
      // Here we need to see if the graph has already been created, and (if so) whether the graph
      // as specified matches what's already been created.
      val graph_dir = s"${praBase}graphs/${graph_name}/"
      val creator = new GraphCreator(praBase, graph_dir, fileUtil)
      if (fileUtil.fileExists(graph_dir)) {
        fileUtil.blockOnFileDeletion(creator.inProgressFile)
        val current_params = parse(fileUtil.readLinesFromFile(creator.paramFile).asScala.mkString("\n"))
        if (params_specified == true && !graphParamsMatch(current_params, params)) {
          println(s"Parameters found in ${creator.paramFile}: ${pretty(render(current_params))}")
          println(s"Parameters specified in spec file: ${pretty(render(params))}")
          println(s"Difference: ${current_params.diff(params)}")
          throw new IllegalStateException("Graph parameters don't match!")
        }
      } else {
        creator.createGraphChiRelationGraph(params)
      }
    }
  }

  // There is a check in the code to make sure that the graph parameters used to create a
  // particular graph in a directory match the parameters you're trying to use with the same graph
  // directory.  But, some things might not matter in that check, like which dense matrices have
  // been created for that graph.  This method specifies which things, exactly, don't matter when
  // comparing two graph parameter specifications.
  def graphParamsMatch(params1: JValue, params2: JValue): Boolean = {
    return params1.removeField(_._1.equals("denser matrices")) ==
      params2.removeField(_._1.equals("denser matrices"))
  }

  def createEmbeddingsIfNecessary(params: JValue) {
    val embeddings = params.filterField(field => field._1.equals("embeddings")).flatMap(_._2 match {
      case JArray(list) => list
      case other => List(other)
    })
    embeddings.filter(_ match {case JString(name) => false; case other => true })
      .par.map(embedding_params => {
        val name = (embedding_params \ "name").extract[String]
        println(s"Checking for embeddings with name ${name}")
        val embeddingsDir = s"${praBase}embeddings/$name/"
        val paramFile = embeddingsDir + "params.json"
        val graph = praBase + "graphs/" + (embedding_params \ "graph").extract[String] + "/"
        val decomposer = new PcaDecomposer(graph, embeddingsDir)
        if (!fileUtil.fileExists(embeddingsDir)) {
          println(s"Creating embeddings with name ${name}")
          val dims = (embedding_params \ "dims").extract[Int]
          decomposer.createPcaRelationEmbeddings(dims)
          val out = fileUtil.getFileWriter(paramFile)
          out.write(pretty(render(embedding_params)))
          out.close
        } else {
          fileUtil.blockOnFileDeletion(decomposer.in_progress_file)
          val current_params = parse(fileUtil.readLinesFromFile(paramFile).asScala.mkString("\n"))
          if (current_params != embedding_params) {
            println(s"Parameters found in ${paramFile}: ${pretty(render(current_params))}")
            println(s"Parameters specified in spec file: ${pretty(render(embedding_params))}")
            println(s"Difference: ${current_params.diff(embedding_params)}")
            throw new IllegalStateException("Embedding parameters don't match!")
          }
        }
    })
  }

  def createSimilarityMatricesIfNecessary(params: JValue) {
    val matrices = params.filterField(field => field._1.equals("similarity matrix")).flatMap(_._2 match {
      case JArray(list) => list
      case other => List(other)
    })
    matrices.filter(_ match {case JString(name) => false; case other => true })
      .par.map(matrixParams => {
        val embeddingsDir = getEmbeddingsDir(matrixParams \ "embeddings")
        val name = (matrixParams \ "name").extract[String]
        val creator = new SimilarityMatrixCreator(embeddingsDir, name)
        if (!fileUtil.fileExists(creator.matrixDir)) {
          creator.createSimilarityMatrix(matrixParams)
        } else {
          fileUtil.blockOnFileDeletion(creator.inProgressFile)
          val current_params = parse(fileUtil.readLinesFromFile(creator.paramFile).asScala.mkString("\n"))
          if (current_params != matrixParams) {
            println(s"Parameters found in ${creator.paramFile}: ${pretty(render(current_params))}")
            println(s"Parameters specified in spec file: ${pretty(render(matrixParams))}")
            println(s"Difference: ${current_params.diff(matrixParams)}")
            throw new IllegalStateException("Similarity matrix parameters don't match!")
          }
        }
    })
  }

  def getEmbeddingsDir(params: JValue): String = {
    params match {
      case JString(path) if (path.startsWith("/")) => path
      case JString(name) => s"${praBase}embeddings/$name/"
      case jval => {
        val name = (jval \ "name").extract[String]
        s"${praBase}embeddings/$name/"
      }
    }
  }

  def createDenserMatricesIfNecessary(params: JValue) {
    val matrices = params.filterField(field => field._1.equals("denser matrices")).flatMap(_._2 match {
      case JArray(list) => list
      case other => List(other)
    })
    matrices.filter(_ match {case JString(name) => false; case other => true })
      .par.map(matrixParams => {
        val graphName = (params \ "graph" \ "name").extract[String]
        val graphDir = s"${praBase}/graphs/${graphName}/"
        val name = (matrixParams \ "name").extract[String]
        val densifier = new GraphDensifier(praBase, graphDir, name)
        if (!fileUtil.fileExists(densifier.matrixDir)) {
          densifier.densifyGraph(matrixParams)
        } else {
          fileUtil.blockOnFileDeletion(densifier.inProgressFile)
          val current_params = parse(fileUtil.readLinesFromFile(densifier.paramFile).asScala.mkString("\n"))
          if (current_params != matrixParams) {
            println(s"Parameters found in ${densifier.paramFile}: ${pretty(render(current_params))}")
            println(s"Parameters specified in spec file: ${pretty(render(matrixParams))}")
            println(s"Difference: ${current_params.diff(matrixParams)}")
            throw new IllegalStateException("Denser matrix parameters don't match!")
          }
        }
    })
  }

  def createSplitIfNecessary(params: JValue) {
    var split_name = ""
    var params_specified = false
    // First, is this just a path, or do the params specify a split name?  If it's a path, we'll
    // just use the path as is.  Otherwise, we have some processing to do.
    params match {
      case JString(path) if (path.startsWith("/")) => {
        if (!fileUtil.fileExists(path)) {
          throw new IllegalStateException("Specified path to split does not exist!")
        }
      }
      case JString(name) => split_name = name
      case jval => {
        split_name = (jval \ "name").extract[String]
        params_specified = true
      }
    }
    if (split_name != "") {
      // Here we need to see if the split has already been created, and (if so) whether the split
      // as specified matches what's already been created.
      val split_dir = s"${praBase}splits/${split_name}/"
      val in_progress_file = SplitCreator.inProgressFile(split_dir)
      val param_file = SplitCreator.paramFile(split_dir)
      if (fileUtil.fileExists(split_dir)) {
        fileUtil.blockOnFileDeletion(in_progress_file)
        if (fileUtil.fileExists(param_file)) {
          val current_params = parse(fileUtil.readLinesFromFile(param_file).asScala.mkString("\n"))
          if (params_specified == true && current_params != params) {
            println(s"Parameters found in ${param_file}: ${pretty(render(current_params))}")
            println(s"Parameters specified in spec file: ${pretty(render(params))}")
            println(s"Difference: ${current_params.diff(params)}")
            throw new IllegalStateException("Split parameters don't match!")
          }
        }
      } else {
        val creator = new SplitCreator(params, praBase, split_dir, fileUtil)
        creator.createSplit()
      }
    }
  }

  def getGraphDirectory(params: JValue): String = {
    (params \ "graph") match {
      case JString(path) if (path.startsWith("/")) => path
      case JString(name) => praBase + "/graphs/" + name
      case jval => praBase + "/graphs/" + (jval \ "name").extract[String]
    }
  }
}

object Driver {

  def initializeGraphParameters(
      graphDirectory: String,
      config: PraConfig.Builder,
      fileUtil: FileUtil = new FileUtil) {
    val dir = fileUtil.addDirectorySeparatorIfNecessary(graphDirectory)
    config.setGraph(dir + "graph_chi" + java.io.File.separator + "edges.tsv")
    println(s"Loading node and edge dictionaries from graph directory: $dir")
    val numShards = fileUtil.readIntegerListFromFile(dir + "num_shards.tsv").get(0)
    config.setNumShards(numShards)
    val nodeDict = new Dictionary()
    nodeDict.setFromReader(fileUtil.getBufferedReader(dir + "node_dict.tsv"))
    config.setNodeDictionary(nodeDict)
    val edgeDict = new Dictionary()
    edgeDict.setFromReader(fileUtil.getBufferedReader(dir + "edge_dict.tsv"))
    config.setEdgeDictionary(edgeDict)
  }

  /**
   * Here we set up the PraConfig items that have to do with the input KB files.  In particular,
   * that means deciding which relations are known to be inverses of each other, which edges
   * should be ignored because using them to predict new relations instances would consitute
   * cheating, and setting the range and domain of a relation to restrict new predictions.
   *
   * Also, if the relations have been embedded into a latent space, we perform a mapping here
   * when deciding which edges to ignore.  This means that each embedding of a KB graph has to
   * have a different directory.
   */
  def parseRelationMetadata(
      directory: String,
      relation: String,
      mode: String,
      builder: PraConfig.Builder,
      outputBase: String,
      fileUtil: FileUtil = new FileUtil) {
    val inverses = Driver.createInverses(directory, builder.edgeDict, fileUtil)
    builder.setRelationInverses(inverses.map(x => (Integer.valueOf(x._1), Integer.valueOf(x._2))).asJava)

    val embeddings = {
      if (directory != null && fileUtil.fileExists(directory + "embeddings.tsv")) {
        fileUtil.readMapListFromTsvFile(directory + "embeddings.tsv").asScala
          .mapValues(_.asScala.toList).toMap
      } else {
        null
      }
    }
    val unallowedEdges = Driver.createUnallowedEdges(relation, inverses, embeddings, builder.edgeDict)
    builder.setUnallowedEdges(unallowedEdges.map(x => Integer.valueOf(x)).asJava)

    if (directory != null && mode != "explore graph" && fileUtil.fileExists(directory + "ranges.tsv")) {
      val ranges = fileUtil.readMapFromTsvFile(directory + "ranges.tsv")
      val range = ranges.get(relation)
      if (range == null) {
        throw new IllegalStateException(
            "You specified a range file, but it doesn't contain an entry for relation " + relation)
      }
      val fixed = range.replace("/", "_")
      val cat_file = directory + "category_instances/" + fixed

      val allowedTargets = fileUtil.readIntegerSetFromFile(cat_file, builder.nodeDict)
      builder.setAllowedTargets(allowedTargets)
    } else {
      val writer = fileUtil.getFileWriter(outputBase + "settings.txt", true)  // true -> append
      writer.write("No range file found! I hope your accept policy is as you want it...\n")
      println("No range file found!")
      writer.close()
    }
  }

  def createUnallowedEdges(
      relation: String,
      inverses: Map[Int, Int],
      embeddings: Map[String, List[String]],
      edgeDict: Dictionary): List[Int] = {
    val unallowedEdges = new mutable.ArrayBuffer[Int]

    // The relation itself is an unallowed edge type.
    val relIndex = edgeDict.getIndex(relation)
    unallowedEdges += relIndex

    // If the relation has an inverse, it's an unallowed edge type.
    inverses.get(relIndex).map(index => unallowedEdges += index)

    val inverse = inverses.get(relIndex) match {
      case Some(index) => edgeDict.getString(index)
      case _ => null
    }

    // And if the relation has an embedding (really a set of cluster ids), those should be
    // added to the unallowed edge type list.
    if (embeddings != null) {
      for (embedded <- embeddings.getOrElse(relation, Nil)) {
        unallowedEdges += edgeDict.getIndex(embedded)
      }
      if (inverse != null) {
        for (embedded <- embeddings.getOrElse(inverse, Nil)) {
          unallowedEdges += edgeDict.getIndex(embedded)
        }
      }
    }
    unallowedEdges.toList
  }

  /**
   * Reads a file containing a mapping between relations and their inverses, and returns the
   * result as a map.
   */
  def createInverses(
      directory: String,
      dict: Dictionary,
      fileUtil: FileUtil = new FileUtil): Map[Int, Int] = {
    val inverses = new mutable.HashMap[Int, Int]
    if (directory == null) {
      inverses.toMap
    } else {
      val filename = directory + "inverses.tsv"
      if (!fileUtil.fileExists(filename)) {
        inverses.toMap
      } else {
        for (line <- fileUtil.readLinesFromFile(filename).asScala) {
          val parts = line.split("\t")
          val rel1 = dict.getIndex(parts(0))
          val rel2 = dict.getIndex(parts(1))
          inverses.put(rel1, rel2)
          // Just for good measure, in case the file only lists each relation once.
          inverses.put(rel2, rel1)
        }
        inverses.toMap
      }
    }
  }

  def initializeSplit(
      splitsDirectory: String,
      relationMetadataDirectory: String,
      relation: String,
      builder: PraConfig.Builder,
      datasetFactory: DatasetFactory,
      fileUtil: FileUtil = new FileUtil) = {
    val fixed = relation.replace("/", "_")
    // We look in the splits directory for a fixed split if we don't find one, we do cross
    // validation.
    if (fileUtil.fileExists(splitsDirectory + fixed)) {
      val training = splitsDirectory + fixed + "/training.tsv"
      val testing = splitsDirectory + fixed + "/testing.tsv"
      builder.setTrainingData(datasetFactory.fromFile(training, builder.nodeDict))
      builder.setTestingData(datasetFactory.fromFile(testing, builder.nodeDict))
      false
    } else {
      if (relationMetadataDirectory == null) {
        throw new IllegalStateException("Must specify a relation metadata directory if you do not "
          + "have a fixed split!")
      }
      builder.setAllData(datasetFactory.fromFile(relationMetadataDirectory + "relations/" + fixed,
                                                 builder.nodeDict))
      val percent_training_file = splitsDirectory + "percent_training.tsv"
      builder.setPercentTraining(fileUtil.readDoubleListFromFile(percent_training_file).get(0))
      true
    }
  }
}
