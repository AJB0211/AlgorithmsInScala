## Functional Algorithms in Scala

Common machine learning algorithms are implemented in base Scala (no Spark or linear algebra libraries.) This implementation is purely functional; no objects are mutated. The purpose of this library is not efficiency but proof of concept therefore runtimes and rates of convergence could be improved. 

Regression algorithms are implemented with gradient descent.


Some features of the Scala programming language utilized:
 - Type variance in fit function
 - Class objects to construct regressors
 - Streams for iterations
 - Zip, map strategy for vectorized operations
 - Monads and monadic maps to handle potential missing values
 - Parallelized matrix multiplication using Scala built in parallel types and aggregate method
    + check out transposeMatDotVec and matDotVec in the Regressor base class!


## Future Work:
  - Use `Data = Array[List[Double]]` for quicker insertion of intercept column
  - Implement dimensionality checks for inputs
  - Implement regularization using L1, L2, elastic traits
  - Use loops to avoid zip operations (requires time complexity analysis)
  - Some way to generalize fit and leave it in the abstract base class
  - Introduce error handling and wrap it instead of exposing type Option[A] as outputs
  - Getters for coefficients
  - toString method for printing coefficents, scores, statistics
  - Add verbose option to return fit information
  - Include statistics (ex. R^2)
  - Fix implicit conversion of boolean to integer and avoid xor
  - Adjust `protected` flags
  - Introduce scala test library
  - Implement unapply methods to extract model to save file
        + Requires method for reimporting from model file


## CHANGE LOG

### V2.0
 - Completion of logistic regressor class (LogReg)
 - Creation of ClassificationTest (not Scala Test library)


### V1.0
 - Linear Regression
 - Regressor Base Class
 - Test cases (not proper Scala Test)
