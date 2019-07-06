abstract class Regressor(val alpha: Double = 0.01,                      // learning rate
                         val coef: Option[Array[Double]] = None,        // Option[ Array of coefficients ], None if fit is not called
                         val score: Option[Double] = None,              // Option [ RMSE ], None if fit not called
                         val intercept: Boolean = true,                 // boolean indicator of if an intercept is used
                         val seed: Int = 0)
{

  type Data = Array[Array[Double]]
  type Weights = Array[Double]


  // Add an intercept to produce a Data s/t each element is an Array(intercept, {coefs})
  def addIntercept(X: Data): Data = X.map(1.0 +: _)

  def fit[U >: Regressor](X: Data,
                          y: Array[Double],
                          iterations: Int,
                          intercept: Boolean,
                          seed: Int = this.seed): U

  def loss(beta: Weights, X: Data, y: Array[Double]): Double

  def descend(beta: Weights, X: Data, y: Array[Double]): Weights

  final def transposeMatDotVec(A: Data, v: Array[Double]): Array[Double] = {
    // Inner product: transpose(A).v
    A.
      zip(v).par
      .aggregate(Array.fill[Double](A.head.length)(0.0))(
        {
          case (acc: Array[Double], (obs: Array[Double],res: Double)) =>
            acc
              .zip(obs.map(_*res))
              .map(t => t._1 + t._2)
        },{
          case (arr1: Array[Double], arr2: Array[Double]) =>
            arr1.zip(arr2)
              .map(t => t._1 + t._2)
        }
      )
  }

  final def matDotVec(A: Data, v: Array[Double]): Array[Double] = {
    // Inner product: A.v
    A.par
      .map( _.zip(v)
        .foldRight(0.0)({case ((f,c),acc) => f*c + acc})
      )
  }.toArray

  // Accessible predict method for returning model predictions
  def predict(X: Data): Option[Array[Double]] = coef match {
    case Some(w) => Some(_predict(w, X))
    case _ => None
  }


  def _predict(w: Weights, X: Data): Array[Double]


  final protected def square(x: Double): Double = x * x
}
