using DFP;
namespace Program
{
    class Program
    {
        decimal[,] d = new decimal[,] // initial state
        {
            {1,0,0},
            {0,1,0},
            {0,0,1}
        };
        public static void Main()
        {
            //input accuracy E
            //input step h
            Func<decimal[], decimal> createFunction = (x) =>(x[0]*x[0]+x[1]-11)*(x[0]*x[0]+x[1]-11)+(x[0]+x[1]*x[1]-7)*(x[0]+x[1]*x[1]-7);//input data
            const decimal accuracy = 1e-6M;
            const decimal anotherAccuracy = 1e-11M;
            const decimal gradientTolerance = 1e-6M;
            // (1 - x[0]) * (1 - x[0]) + 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0])
            // (x[0]*x[0]+x[1]-11)*(x[0]*x[0]+x[1]-11)+(x[0]+x[1]*x[1]-7)*(x[0]+x[1]*x[1]-7)   Himmelblau function
            // 0.26*(x[0]*x[0]+x[1]*x[1]) - 0.48*x[0]*x[1]
            // 2*x[0]*x[0]+x[1]*x[0]+x[1]*x[1]
            decimal[,] x0 = new decimal[1, 2] { { -3.5M, -1 } };
            decimal[,] result = DavidonFletcherPowell.Minimize(createFunction, x0, accuracy, gradientTolerance, anotherAccuracy);

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