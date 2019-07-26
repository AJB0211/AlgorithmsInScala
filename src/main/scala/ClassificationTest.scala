import scala.util.Random



object ClassificationTest extends App{
  // Offset parameter adjusts separation of means for random distributions
  // These distributions indicate the distance between the two test classes in feature space
  // Reducing the offset will increase error in regression once the distributions begin to overlap
  val offset: Double = 1.0
  val rnd: Random = new Random(6)
  val numPoints: Int = 1000

  // Initialize two classes of points, true and false about axis x=0
  val trues: Array[Array[Double]]  = Array.fill[Array[Double]](numPoints)(
                                Array.fill[Double](1)(rnd.nextGaussian + offset))
  val falses: Array[Array[Double]]  = Array.fill[Array[Double]](numPoints)(
                                Array.fill[Double](1)(rnd.nextGaussian - offset))

  // Append arrays and make labels
  val X: Array[Array[Double]] = trues ++ falses
  val y: Array[Double] = Array.fill[Double](trues.length)(1.0) ++ Array.fill[Double](falses.length)(0.0)

  val reg: LogReg = LogReg(0.01).fit(X,y, iterations = 5000)

  println
  reg.predict(X).get.zip(y).zip(reg._predict(reg.coef.get,X)).foreach(println)
  println("prediction actual logodds")
  println
  println("Coefficients:")
  reg.coef.get.foreach(out => println(f"\t$out%.5f"))
  println
  println(f"Loss:     ${reg.score.getOrElse(666.0)}%.3f")
  println(f"Accuracy: ${reg.accuracy(X,y).getOrElse(666.0)}%.3f")
  println


}
