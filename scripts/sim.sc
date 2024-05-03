//> using scala 2.13

import java.io.FileOutputStream
import java.util.zip.GZIPOutputStream
import java.io.OutputStream
import java.io.File
import java.io.PrintWriter
import java.nio.charset.StandardCharsets
import scala.util.Using
import scala.io.Source

final case class TermWithIC(term: String)(val ic: Double)

val annotationsPath = args(0)
val subsumptionsPath = args(1)
val outputPath = args(2)

val annotations = Using.resource(Source.fromFile(annotationsPath, "utf-8"))(
  _.getLines()
    .map { line =>
      val items = line.split("\t", -1)
      items(0) -> items(1)
    }
    .toList
    .groupMap(_._1)(_._2)
    .map(kv => kv._1 -> kv._2.toSet)
)
val items = annotations.keySet
val corpusSize = items.size
println(s"Corpus size: $corpusSize")
val maxIC = -Math.log(1.0 / corpusSize)
val scale = 100.0 / maxIC
val usedTerms = annotations.values.flatten.to(Set)
val subsumptions = Using.resource(Source.fromFile(subsumptionsPath, "utf-8"))(
  _.getLines()
    .map { line =>
      val items = line.split("\t", -1)
      items(0) -> items(1)
    }
    .filter(pair => usedTerms(pair._1))
    .toList
    .groupMap(_._1)(_._2)
    .map(kv => kv._1 -> kv._2.to(Set))
)
val classes = subsumptions.values.flatten.to(Set)
val expandedAnnotations = annotations.map { case (item, terms) =>
  item -> terms.flatMap(subsumptions)
}
val classToIC = (for {
  (item, classes) <- expandedAnnotations.toSeq
  cls <- classes
} yield cls -> item)
  .groupMap(_._1)(_._2)
  .map { case (cls, items) =>
    cls -> (-Math.log(items.to(Set).size * 1.0 / corpusSize) * scale)
  }
val icAnnotations = for {
  (item, classes) <- expandedAnnotations
} yield item -> classes.map(cls => TermWithIC(cls)(classToIC(cls)))
Using.resource(
  new PrintWriter(
    new GZIPOutputStream(new FileOutputStream(new File(outputPath))),
    true
  )
) { writer =>
  for {
    (a, aClasses) <- icAnnotations
    (b, bClasses) <- icAnnotations
    if a < b
  } {
    val intersection = aClasses.intersect(bClasses)
    val maxIC =
      if (intersection.nonEmpty) intersection.maxBy(_.ic).ic
      else 0.0
    val union = aClasses.union(bClasses)
    val jaccard = intersection.size.toDouble / union.size.toDouble * 100.0
    val simGIC =
      intersection.iterator
        .map(_.ic)
        .sum
        .toDouble /
        union.iterator
          .map(_.ic)
          .sum
          .toDouble * 100.0
    writer.println(s"$a\t$b\t$maxIC\t$jaccard\t$simGIC")
  }
}
