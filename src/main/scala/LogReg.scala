
import scala.util.Random
import scala.math.{exp, log, round}


class LogReg(override val alpha: Double = 0.01,                      // learning rate
        override val coef: Option[Array[Double]] = None,        // Option[ Array of coefficients ], None if fit is not called
        override val score: Option[Double] = None,              // Option [ Loss ], None if fit not called
        override val intercept: Boolean = true,                 // boolean indicator of if an intercept is used
        override val seed: Int = 0)                             // set seed for reproducibility, passed to random initializer
        extends Regressor(alpha,coef,score,intercept,seed) {

  // figure out how to place this in Regressor
  // fit call self constructor for derived types
  override def fit[U >: Regressor]( X: Data,
                                    y: Array[Double],
                                    iterations: Int = 1000,
                                    intercept: Boolean = true,
                                    seed: Int = this.seed): U = {

    // Create local data array if intercept is used
    val _X: Data = if (intercept) addIntercept(X) else X

    val initializer: Random = new Random()
    initializer.setSeed(seed)

    val init: Weights = Array.fill[Double](_X.head.length)(initializer.nextDouble)

    def fitStream(it: Weights): Stream[Weights] = it #:: fitStream(descend(it, _X, y))

    def beta: Weights = fitStream(init).drop(iterations).head

    new LogReg(alpha, Some(beta), Some(loss(beta, X, y)), intercept, seed)
  }

  override def loss(beta: Weights, X: Data, y: Array[Double]): Double = {
    // Computes positive log loss, not loss or negative log los
    y.zip(_predict(beta,X))
      .map{case (y,p) => y*log(p) + (1-y)*log(1-p)}
      .sum / X.length
  }

  override def predict(X:Data): Option[Array[Double]] = coef.map( w => _predict(w,X).map(round(_).toDouble))

  override def _predict(w: Weights, X: Data): Array[Double] = matDotVec(X,w).map(sigmoid)

  override def grad(w: Weights, X: Data, y: Array[Double]): Weights = {
    // - x.T (y - _predict)
    transposeMatDotVec(X,
      y.zip(_predict(w,X))
          .map(p => p._2 - p._1)
      )
  }

  // Figure out how to get implicits to work and modify accuracy
//  implicit def boolToInt(b: Boolean): Numeric[Boolean] = if (b) 1 else 0

  def accuracy(X: Data, y: Array[Double]): Option[Double] = predict(X).map(
    preds => preds.zip(y)
      .map(p => 1 - p._1.toInt^p._2.toInt)
      .sum / y.length.toDouble
  )

  final def sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-1.0*x))
}




  object LogReg {
    def apply(alpha: Double): LogReg = new LogReg(alpha)
    // def unapply
}

