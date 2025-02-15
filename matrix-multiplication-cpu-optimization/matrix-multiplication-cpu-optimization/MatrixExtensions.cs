using System.Text;

namespace matrix_multiplication_cpu_optimization
{
    /// <summary>
    /// Helper functions for working with matrices
    /// </summary>
    public static class MatrixExtensions
    {
        /// <summary>
        /// Calculate sum of two matrices
        /// </summary>
        /// <param name="a">First matrix</param>
        /// <param name="b">Second matrix</param>
        /// <returns>Result of summation</returns>
        public static double[,] Add(this double[,] a, double[,] b)
        {
            var (rows, cols) = (a.GetLength(0), a.GetLength(1));
            var result = new double[rows, cols];
            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    result[i, j] = a[i, j] + b[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Calculate difference of two matrices
        /// </summary>
        /// <param name="a">First matrix</param>
        /// <param name="b">Second matrix</param>
        /// <returns>Result of subtraction</returns>
        public static double[,] Subtract(this double[,] a, double[,] b)
        {
            var (rows, cols) = (a.GetLength(0), a.GetLength(1));
            var result = new double[rows, cols];
            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    result[i, j] = a[i, j] - b[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Convert matrix to string for printing
        /// </summary>
        /// <param name="x">Matrix to convert</param>
        /// <returns>Nice-looking string</returns>
        public static string Format(this double[,] x)
        {
            var (rows, cols) = (x.GetLength(0), x.GetLength(1));
            var builder = new StringBuilder();
            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    builder.Append(x[i, j]);
                    if (j != cols - 1)
                    {
                        builder.Append(" ");
                    }
                }

                if (i != rows - 1)
                {
                    builder.Append('\n');
                }
            }

            return builder.ToString();
        }
    }
}
