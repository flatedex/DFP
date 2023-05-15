using System;

namespace DFP
{
	public class DavidonFletcherPowell
	{
		// функция поиска шага t
		private static decimal LineSearch(Func<decimal[], decimal> function, decimal[,] point, decimal[] direction, decimal epsilon)
		{
			decimal alpha = 1.0M;
			decimal loBound = 0.0M;
			decimal hiBound = Decimal.MaxValue;
			decimal[] newPoint = FromMatrixToVector(point);
			while (true)
			{
				decimal[] candidate = Add(newPoint, Scale(alpha, direction));
				decimal functionValue = function(candidate);
				decimal loValue = function(Add(newPoint, Scale(loBound, direction)));
				if (functionValue > loValue + epsilon)
				{
					hiBound = alpha;
					alpha = (loBound + hiBound) / 2.0M;
				}
				else
				{
					decimal hiValue = function(Add(newPoint, Scale(hiBound, direction)));
					if (functionValue > hiValue + epsilon)
					{
						loBound = alpha;
						alpha = (loBound + hiBound) / 2.0M;
					}
					else
					{
						return alpha;
					}
				}
			}
		}

		// функция нахождения градиента
		private static decimal[] Gradient(Func<decimal[], decimal> function, decimal[,] point, decimal epsilon)
		{
			int n = point.Length;
			decimal[] newPoint = FromMatrixToVector(point);
			decimal[] result = new decimal[n];
			for (int i = 0; i < n; i++)
			{
				decimal[] perturbation = new decimal[n];
				perturbation[i] = epsilon;
				decimal v1 = function(Add(newPoint, perturbation));
				decimal v2 = function(Subtract(newPoint, perturbation));
				decimal v3 = v1 - v2;
                //result[i] = (function(Add(newPoint, perturbation)) - function(Subtract(newPoint, perturbation))) / (2 * epsilon);
                result[i] = v3 / (2 * epsilon);
			}
			return result;
		}

		// функция суммирования векторов
		private static decimal[] Add(decimal[] vector1, decimal[] vector2)
		{
			int n = vector1.Length;
			decimal[] result = new decimal[n];
			for (int i = 0; i < n; i++)
			{
				result[i] = vector1[i] + vector2[i];
			}
			return result;
		}
		private static decimal[] Add(decimal[,] vector1, decimal[] vector2)
		{
			int n = vector1.Length;
			decimal[] result = new decimal[n];
			decimal[] newVector = FromMatrixToVector(vector1);
			for (int i = 0; i < n; i++)
			{
				result[i] = newVector[i] + vector2[i];
			}
			return result;
		}
		// функция вычитания векторов
		private static decimal[] Subtract(decimal[] vector1, decimal[] vector2)
		{
			int n = vector1.Length;
			decimal[] result = new decimal[n];
			for (int i = 0; i < n; i++)
			{
				result[i] = vector1[i] - vector2[i];
			}
			return result;
		}
		private static decimal[] Subtract(decimal[] vector1, decimal[,] vector2)
		{
			int n = vector1.Length;
			decimal[] result = new decimal[n];
			decimal[] newVector = FromMatrixToVector(vector2);
			for (int i = 0; i < n; i++)
			{
				result[i] = vector1[i] - newVector[i];
			}
			return result;
		}

		// функция умножения вектора на скаляр
		private static decimal[] Scale(decimal scalar, decimal[] vector)
		{
			int n = vector.Length;
			decimal[] result = new decimal[n];
			for (int i = 0; i < n; i++)
			{
				result[i] = scalar * vector[i];
			}
			return result;
		}

		//функция умножения матрицы на скаляр
		private static decimal[,] ScaleMatrix(decimal scalar, decimal[,] matrix)
		{
			int n = matrix.GetLength(0);
			decimal[,] result = new decimal[n, n];
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					result[i, j] = matrix[i, j] * scalar;
				}
			}
			return result;
		}

		// функция умножения матрицы на вектор
		private static decimal[] MatrixOnVectorDotProduct(decimal[] vector, decimal[,] matrix)
		{
			int n = vector.Length;
			int m = matrix.GetLength(1);
			decimal[] result = new decimal[n];
			for (int i = 0; i < n; i++)
			{
				decimal sum = 0.0M;
				for (int j = 0; j < m; j++)
				{
					sum += vector[i] * matrix[i, j];
				}
				result[i] = sum;
			}
			return result;
		}

		// функция умножения матрицы на матрицу
		private static decimal[,] MatrixOnMatrixDotProduct(decimal[,] matrix1, decimal[,] matrix2)
		{
			int n = matrix1.GetLength(0);
			int m = matrix1.GetLength(1);
			int p = matrix2.GetLength(1);
			decimal[,] result = new decimal[n, p];
			for (int j = 0; j < p; j++)
			{
				for (int i = 0; i < n; i++)
				{
					decimal sum = 0.0M;
					for (int k = 0; k < m; k++)
					{
						sum += matrix1[i, k] * matrix2[k, j];
					}
					result[i, j] = sum;
				}
			}
			return result;
		}
           
		// функция скалярного произведения векторов
		private static decimal VectorOnVectorDotProduct(decimal[] vector1, decimal[,] vector2)
		{
			int n = vector1.GetLength(0);
			int p = vector2.GetLength(1);
			decimal result = 0;
			for (int i = 0; i < p; i++)
			{
				result += vector1[i] * vector2[0, i];
			}
			return result;
		}
		// функция вычитания матриц
		private static decimal[,] MatrixSubtraction(decimal[,] matrix1, decimal[,] matrix2)
		{
			int n = matrix1.GetLength(0);
			int m = matrix1.GetLength(1);
			decimal[,] result = new decimal[n, m];
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < m; j++)
				{
					result[i, j] = matrix1[i, j] - matrix2[i, j];
				}
			}
			return result;
		}
		private static decimal[,] SumOfMatrices(decimal[,] matrix1, decimal[,] matrix2)
		{
			int n = matrix1.GetLength(0);
			int m = matrix1.GetLength(1);
			decimal[,] result = new decimal[n, m];
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < m; j++)
				{
					result[i, j] = matrix1[i, j] + matrix2[i, j];
				}
			}
			return result;
		}
		// функция превращения вектора [vector.Length] в матрицу [vector.Length, 1]
		private static decimal[,] MatrixTranspose(decimal[] vector)
		{
			int n = vector.Length;
			decimal[,] result = new decimal[n, 1];
			for (int i = 0; i < n; i++)
			{
				result[i, 0] = vector[i];
			}
			return result;
		}
		// фукнция перевода вектора [vector.Length] в матрицу [1, vector.Length]
		private static decimal[,] FromVectorToMatrix(decimal[] vector)
		{
			int n = vector.Length;
			decimal[,] result = new decimal[1, n];
			for (int i = 0; i < n; i++)
			{
				result[0, i] = vector[i];
			}
			return result;
		}
		private static decimal[] FromMatrixToVector(decimal[,] matrix)
		{
			int n = matrix.GetLength(1);
			decimal[] result = new decimal[n];
			for (int i = 0; i < n; i++)
			{
				result[i] = matrix[0, i];
			}
			return result;
		}
		// функция создания единичной матрицы размерностью n
		private static decimal[,] IdentityMatrix(int n)
		{
			decimal[,] result = new decimal[n, n];
			for (int i = 0; i < n; i++)
			{
				result[i, i] = 1.0M;
			}
			return result;
		}
		// функция для вычисления нормы вектора
		private static decimal Norm(decimal[] x)
		{
			decimal norm = 0;
			for (int i = 0; i < x.Length; i++)
			{
				norm += x[i] * x[i];
			}
			return Sqrt(norm);
		}
        public static decimal Abs(decimal x) {
            if (x <= 0.0M) {
                return -x;
            }
            return x;
        }
        public static decimal Sqrt(decimal x, decimal epsilon = 0.0M) {
            if (x < 0.0M)
                throw new OverflowException("Cannot calculate square root from a negative number");
            //initial approximation
            decimal current = (decimal)Math.Sqrt((double)x), previous;
            do {
                previous = current;
                if (previous == 0.0M)
                    return 0.0M;
                current = (previous + x / previous) * 0.5M;
            } while (Abs(previous - current) > epsilon);
            return current;
        }
        public static decimal[,] CreateMatrixOfZeros(int n, int m){
            decimal[,] matrix = new decimal[n, m];
            for(int i = 0; i < n; i++){
                for(int j = 0; j < m; j++){
                    matrix[i,j] = 0;
                }
            }
            return matrix;
        }
        private static decimal[,] OuterProduct(decimal[] vector1, decimal[] vector2){
            int n = vector1.Length;
            int m = vector2.Length;
            decimal[,] result = new decimal[n,m];
            for(int i = 0; i < n; i++){
                for(int j = 0; j < m; j++){
                    result[i,j] = vector1[i]*vector2[j];
                }
            }
            return result;
        }
        public static decimal[,] Minimize(Func<decimal[], decimal> f, decimal[,] x0, decimal accuracy, decimal _gradientTolerance, decimal _newAccuracy)
		{
			// Начальное приближение
			decimal[,] x = x0;
			// Инициализируем единичную матрицу
			decimal[,] A0 = IdentityMatrix(x.Length);
			// Устанавливаем максимальное число итераций
			int maxIterations = 1000;
			// Устанавливаем пороговое значение для нормы градиента, т.н. точность
			decimal gradientTolerance = _gradientTolerance; // recomended 1e-6
			decimal newAccuracy = _newAccuracy; // recomended 1e-3

			int k;
			for (k = 0; k < maxIterations; k++)
			{
				// Вычисляем градиент в точке x
				decimal[] grad = Gradient(f, x, gradientTolerance);

				// Проверяем, достигнута ли заданная точность
				if (Norm(grad) < newAccuracy)
				{
					//System.Console.WriteLine(k.ToString());
					return x;
				}

				// Вычисляем направление спуска
				decimal[] d = MatrixOnVectorDotProduct(grad, A0);

				// Вычисляем длину шага по направлению градиента
				decimal t = 0;
				try{
					t = LineSearch(f, x, d, accuracy);
                    if(t < newAccuracy) return x;
				}
				catch(Exception e){
					//Console.WriteLine(e);
					t = 0.0001M; // magic number if something goes wrong
				}

				// Выполняем шаг оптимизации
				decimal[] xNew = Add(x, Scale(t, d));
				decimal[,] xNewMatrix = FromVectorToMatrix(xNew);

				// Вычисляем разность градиентов в точках xNew и x
				decimal[] gradNew = Gradient(f, xNewMatrix, gradientTolerance);
				decimal[] deltaGrad = Subtract(gradNew, grad);

				// Вычисляем разность аргументов в точках xNew и x
				decimal[] deltaX = Subtract(xNew, x);

                decimal[,] numerator1 = CreateMatrixOfZeros(A0.GetLength(0), A0.GetLength(1));
                decimal[,] numerator2 = CreateMatrixOfZeros(A0.GetLength(0), A0.GetLength(1));
				numerator1 = OuterProduct(deltaX, deltaX);
				numerator2 = MatrixOnMatrixDotProduct(MatrixOnMatrixDotProduct(MatrixOnMatrixDotProduct(A0, MatrixTranspose(deltaGrad)), FromVectorToMatrix(deltaGrad)), A0);
				decimal denominetor1 = VectorOnVectorDotProduct(deltaGrad, MatrixTranspose(deltaX));
				decimal denominetor2 = VectorOnVectorDotProduct(deltaGrad, MatrixOnMatrixDotProduct(MatrixTranspose(deltaGrad), A0));

				// Вычисляем матрицу Ak+1
				decimal[,] A = CreateMatrixOfZeros(A0.GetLength(0), A0.GetLength(0));
                A = SumOfMatrices(A, MatrixSubtraction(ScaleMatrix(1 / denominetor1, numerator1), ScaleMatrix(1 / denominetor2, numerator2)));

				// Обновляем x и H
				x = xNewMatrix;
				A0 = A;
			}
			// Если не удалось достичь заданной точности за максимальное число итераций,
			// возвращаем последнее

			return x;
		}
	}
}