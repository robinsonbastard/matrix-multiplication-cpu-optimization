namespace matrix_multiplication_cpu_optimization;

/// <summary>
/// Solver that multiplies matrices by definition
/// </summary>
public class NaiveSolver : ISolver
{
    /// <inheritdoc/>
    public double[,] Multiply(double[,] a, double[,] b)
    {
        var rows = a.GetLength(0);
        var columns = b.GetLength(1);
        var length = a.GetLength(1);

        var result = new double[rows, columns];
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                for (var i = 0; i < length; i++)
                {
                    result[row, column] += a[row, i] * b[i, column];
                }
            }
        }

        return result;
    }
}