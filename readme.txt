The Davidon–Fletcher–Powell formula finds the solution to the secant equation that is closest to the current estimate and satisfies the curvature condition.
It was the first quasi-Newton method to generalize the secant method to a multidimensional problem. 

Here are some functions for testing:

Rosenbrock function. Answer must be in f(1, 1) = 0
(1 - x[0]) * (1 - x[0]) + 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0])  

Himmelblau function. Answer must be in f(3, 2) = 0
(x[0]*x[0]+x[1]-11)*(x[0]*x[0]+x[1]-11)+(x[0]+x[1]*x[1]-7)*(x[0]+x[1]*x[1]-7)   

Matyas function. Answer must be in f(0, 0) = 0
0.26*(x[0]*x[0]+x[1]*x[1]) - 0.48*x[0]*x[1]                                   
