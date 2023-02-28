using System;

class DavidonFletcherPowell
{
    // функция вычисления длины вектора
    private static double Magnitude(double[] vector)
    {
        double result = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            result += vector[i];
        }

        return Math.Sqrt(result);
    }
    // функция вычисления обратного вектора
    private static double[] Negate(double[] vector)
    {
        double[] result = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = -vector[i];
        }
        return result;
    }

    // функция поиска шага
    private static double LineSearch(Func<double[], double> function, double[] point, double[] direction, double epsilon)
    {
        double alpha = 1.0;
        double loBound = 0.0;
        double hiBound = Double.PositiveInfinity;
        while (true)
        {
            double[] candidate = Add(point, Scale(alpha, direction));
            double functionValue = function(candidate);
            double loValue = function(Add(point, Scale(loBound, direction)));
            if (functionValue > loValue + epsilon)
            {
                hiBound = alpha;
                alpha = (loBound + hiBound) / 2.0;
            }
            else
            {
                double hiValue = function(Add(point, Scale(hiBound, direction)));
                if (functionValue > hiValue + epsilon)
                {
                    loBound = alpha;
                    alpha = (loBound + hiBound) / 2.0;
                }
                else
                {
                    return alpha;
                }
            }
        }
    }

    // функция градиента
    private static double[] Gradient(Func<double[], double> function, double[] point, double epsilon)
    {
        int n = point.Length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++)
        {
            double[] perturbation = new double[n];
            perturbation[i] = epsilon;
            result[i] = (function(Add(point, perturbation)) - function(Subtract(point, perturbation))) / (2 * epsilon);
        }
        return result;
    }

    // функция суммирования векторов
    private static double[] Add(double[] vector1, double[] vector2)
    {
        int n = vector1.Length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++)
        {
            result[i] = vector1[i] + vector2[i];
        }
        return result;
    }

    // функция вычитания векторов
    private static double[] Subtract(double[] vector1, double[] vector2)
    {
        int n = vector1.Length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++)
        {
            result[i] = vector1[i] - vector2[i];
        }
        return result;
    }

    // функция умножения вектора на скаляр
    private static double[] Scale(double scalar, double[] vector)
    {
        int n = vector.Length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++)
        {
            result[i] = scalar * vector[i];
        }
        return result;
    }

    // функция умножения матрицы на вектор
    private static double[] MatrixOnVectorDotProduct(double[] vector, double[,] matrix)
    {
        int n = vector.Length;
        int m = matrix.GetLength(1);
        double[] result = new double[m];
        for (int j = 0; j < m; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < n; i++)
            {
                sum += vector[i] * matrix[i, j];
            }
            result[j] = sum;
        }
        return result;
    }

    // функция умножения матрицы на матрицу
    private static double[,] MatrixOnMatrixDotProduct(double[,] matrix1, double[,] matrix2)
    {
        int n = matrix1.GetLength(0);
        int m = matrix1.GetLength(1);
        int p = matrix2.GetLength(1);
        double[,] result = new double[n, p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int k = 0; k < m; k++)
                {
                    sum += matrix1[i, k] * matrix2[k, j];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }
    private static double VectorOnVectorDotProduct(double[] vector1, double[] vector2)
    {
        int n = vector1.GetLength(0);
        int p = vector2.GetLength(1);
        double result = 0;
        for (int i = 0; i < p; i++)
        {
            result += vector1[i] * vector2[i];
        }
        return result;
    }

    // функция внешнего произведения векторов
    private static double[,] MatrixOuterProduct(double[] vector1, double[] vector2)
    {
        int n = vector1.Length;
        double[,] result = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = vector1[i] * vector2[j];
            }
        }
        return result;
    }

    // функция сложения матриц
    private static double[,] MatrixAddition(double[,] matrix1, double[,] matrix2)
    {
        int n = matrix1.GetLength(0);
        int m = matrix1.GetLength(1);
        double[,] result = new double[n, m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                result[i, j] = matrix1[i, j] + matrix2[i, j];
            }
        }
        return result;
    }
    // функция вычитания матриц
    private static double[,] MatrixSubtraction(double[,] matrix1, double[,] matrix2)
    {
        int n = matrix1.GetLength(0);
        int m = matrix1.GetLength(1);
        double[,] result = new double[n, m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                result[i, j] = matrix1[i, j] - matrix2[i, j];
            }
        }
        return result;
    }
    // функция транспонирования матрицы
    private static double[,] MatrixTranspose(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        int m = matrix.GetLength(1);
        double[,] result = new double[m, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                result[j, i] = matrix[i, j];
            }
        }
        return result;
    }
    private static double[,] IdentityMatrix(int n)
    {
        double[,] result = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            result[i, i] = 1.0;
        }
        return result;
    }
    public static double[] Minimize(Func<double[], double> f, double[] x0)
    {
        // Начальное приближение
        double[] x = x0;
        // Инициализируем единичную матрицу
        double[,] H = IdentityMatrix(x.Length);
        // Устанавливаем максимальное число итераций
        int maxIterations = 100;
        // Устанавливаем пороговое значение для нормы градиента
        double gradientTolerance = 1e-6;

        for (int k = 0; k < maxIterations; k++)
        {
            // Вычисляем градиент в точке x
            double[] grad = Gradient(f, x, gradientTolerance);

            // Проверяем, достигнута ли заданная точность
            if (Norm(grad) < gradientTolerance)
            {
                return x;
            }

            // Вычисляем направление спуска
            double[] p = MatrixOnVectorDotProduct(grad, H);

            // Вычисляем длину шага по направлению p
            double alpha = LineSearch(f, x, p, gradientTolerance);

            // Выполняем шаг оптимизации
            double[] xNew = Add(x, Scale(alpha, p));

            // Вычисляем разность градиентов в точках xNew и x
            double[] gradNew = Gradient(f, xNew, gradientTolerance);
            double[] deltaGrad = Subtract(gradNew, grad);

            // Вычисляем разность аргументов в точках xNew и x
            double[] deltaX = Subtract(xNew, x);

            // Вычисляем матрицу Bk+1
            double[,] B = MatrixAddition(H, MatrixOuterProduct(deltaGrad, deltaGrad) / VectorOnVectorDotProduct(deltaGrad, deltaX)
                - MatrixOuterProduct(MatrixOnVectorDotProduct(deltaGrad, H), MatrixOnVectorDotProduct(deltaGrad, H)) / VectorOnVectorDotProduct(MatrixOnVectorDotProduct(deltaGrad, H), deltaGrad));

            // Обновляем x и H
            x = xNew;
            H = B;
        }

        // Если не удалось достичь заданной точности за максимальное число итераций,
        // возвращаем последнее

        return x;
    }
    // функция для вычисления нормы вектора
    private static double Norm(double[] x)
    {
        double norm = 0;
        for (int i = 0; i < x.Length; i++)
        {
            norm += x[i] * x[i];
        }
        return Math.Sqrt(norm);
    }
}


