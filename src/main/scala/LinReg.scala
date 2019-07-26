import math.sqrt
import scala.util.Random


// Fix member definitions with inheritance
class LinReg(override val alpha: Double = 0.01,                      // learning rate
             override val coef: Option[Array[Double]] = None,        // Option[ Array of coefficients ], None if fit is not called
             override val score: Option[Double] = None,              // Option [ RMSE ], None if fit not called
             override val intercept: Boolean = true,                 // boolean indicator of if an intercept is used
             override val seed: Int = 0)                             // set seed for reproducibility, passed to random initializer
 extends Regressor(alpha,coef,score,intercept,seed){


  // figure out how to place this in Regressor
  // fit call self constructor for derived types
//  override def fit[U >: Regressor](
//          X: Data,
//          y: Array[Double],
//          iterations: Int = 1000,
//          intercept: Boolean = true,
//          seed: Int = this.seed): U = {
//
//    // Create local data array if intercept is used
//    val _X: Data = if (intercept) addIntercept(X) else X
//
//    val initializer: Random = new Random()
//    initializer.setSeed(seed)
//
//    val init: Weights = Array.fill[Double](_X.head.length)(initializer.nextDouble)
//
//    def fitStream(it: Weights): Stream[Weights] = it #:: fitStream(descend(it,_X,y))
//
//    lazy val beta: Weights = fitStream(init).drop(iterations).head
//
//    new LinReg(alpha, Some(beta), Some(loss(beta,X,y)), intercept, seed)
//
//  }

   def fit(
            X: Data,
            y: Array[Double],
            iterations: Int = 1000,
            intercept: Boolean = true,
            seed: Int = this.seed): LinReg = {

    // Create local data array if intercept is used
    val _X: Data = if (intercept) addIntercept(X) else X

    val initializer: Random = new Random()
    initializer.setSeed(seed)

    val init: Weights = Array.fill[Double](_X.head.length)(initializer.nextDouble)

    def fitStream(it: Weights): Stream[Weights] = it #:: fitStream(descend(it,_X,y))

    lazy val beta: Weights = fitStream(init).drop(iterations).head

    new LinReg(alpha, Some(beta), Some(loss(beta,X,y)), intercept, seed)

  }

  override def loss(beta: Weights, X: Data, y: Array[Double]): Double = {
    sqrt(
      _predict(beta, X)
        .zip(y).par
        .map { p => square(p._2 - p._1) }
        .fold(0.0)(_ + _) / X.length
    )
  }

  override def grad(w: Weights, X: Data, y: Array[Double]): Array[Double] = {
    // Internal method for computation of the gradient
    val c: Double = -2.0 / X.length                             // -2/Number of observations , coefficient outside summation in descent
    transposeMatDotVec(X,_residuals(w, X, y)).map(_*c)
  }

  protected def _residuals(w: Weights, X: Data, y: Array[Double]): Array[Double] = {
    y.zip(_predict(w, X))
      .map { case (yTrue, yPred) => yTrue - yPred }
  }

  override def _predict(w: Weights, X: Data): Array[Double] = matDotVec(X,w)

}


object LinReg {
  def apply(alpha: Double = 0.1): LinReg =  new LinReg(alpha)
  // def unapply => return coefficient vector / output vector to text

}
