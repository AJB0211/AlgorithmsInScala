import scala.util.Random



object RegressionTest extends App {
  val numPoints: Int = 5000
  val slope: Array[Double] = Array(scala.math.Pi, 2.0)
  val intercept: Double = 1.0
  val errorDeviation: Double = 0.01


  val rnd: Random = new Random(0)
  val X: Array[Array[Double]] = Array.fill[Array[Double]](numPoints)(Array.fill[Double](slope.length)(Random.nextDouble))


  val y: Array[Double] = {
    X.map(_.zip(slope).map {
      case (x, m) => x * m
    }.sum + intercept + +errorDeviation * rnd.nextGaussian)
  }


//  println(X.length)
//  println(X.head.length)
//
//  println(y.length)


  val reg: Regressor = {
    LinReg(0.1)
      .fit(X, y, iterations = 10000)
  }

  println
  reg.coef.get.foreach(println)
  println
  println(reg.score.get)

}




//object LinRegTest extends LinReg {
//  def apply(alpha: Double): LinReg = {
//    val out = new LinReg(alpha)
//
//    out.transposeMatDotVec(Array(Array(1,0,0),Array(0,1,0),Array(0,0,1)),Array(1,2,3)).foreach(println)
//
//    println("*" * 50)
//
//    // Array(70.0,80.0,90.0)
//    out.transposeMatDotVec(Array(Array(1,2,3),Array(4,5,6),Array(7,8,9),Array(10,11,12)),Array(1,2,3,4)).foreach(println)
//
//    println("*" * 50)
//
//    // Array(1,2,3)
//    out.matDotVec(Array(Array(1,0,0),Array(0,1,0),Array(0,0,1)),Array(1,2,3)).foreach(println)
//
//    println("*" * 50)
//
//    // Array(14.0,32,0,50.0,68.0)
//    out.matDotVec(Array(Array(1,2,3),Array(4,5,6),Array(7,8,9),Array(10,11,12)),Array(1,2,3)).foreach(println)
//
//    println("*" * 50)
//
//    out
//  }
//  // def unapply => return coefficient vector / output vector to text
//
//
//}