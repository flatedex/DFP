namespace Program
{
    class Program
    {
        double[,] d = new double[,] // initial state
        {
            {1,0,0},
            {0,1,0},
            {0,0,1}
        };
        public static void Main()
        {
            //input accuracy E
            //input step h
            Func<double[], double> createFunction = (x) => 0.26 * (x[0] * x[0] + x[1] * x[1]) - 0.48 * x[0] * x[1];//input data
            // (1 - x[0]) * (1 - x[0]) + 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0])
            // x[0]*x[0]+x[1]-11)*(x[0]*x[0]+x[1]-11)+(x[0]+x[1]*x[1]-7)*(x[0]+x[1]*x[1]-7   Himmelblau function
            // 0.26*(x[0]*x[0]+x[1]*x[1]) - 0.48*x[0]*x[1]
            double[,] x0 = new double[2, 1] { { 0.5 }, { 1 } };
            double[,] result = DavidonFletcherPowell.Minimize(createFunction, x0);

            for (int i = 0; i < result.GetLength(0); i++)
            {
                System.Console.WriteLine();
                for (int j = 0; j < result.GetLength(1); j++)
                {
                    System.Console.Write("   " + result[i, j]);
                }
            }
        }
    }
}