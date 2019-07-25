import scala.util.Random



object ClassificationTest extends App{
  val offset: Double = 1.0
  val rnd: Random = new Random(0)

  val trues: Array[Double]  = Array.fill[Double](50)(rnd.nextGaussian).map( _ + offset)
  val falses: Array[Double] = Array.fill[Double](50)(rnd.nextGaussian).map( _ - offset)

  val X: Array[Array[Double]] = Array( trues ++ falses)
  val y: Array[Double] = Array.fill[Double](50)(1.0) ++ Array.fill[Double](50)(0.0)

  val reg: LogReg = LogReg(0.1).fit(X,y, iterations = 10000)

  println
  println(reg.accuracy(X,y))
  println


}
