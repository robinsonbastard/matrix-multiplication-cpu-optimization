namespace matrix_multiplication_cpu_optimization;

/// <summary>
/// Generator for creating random matrices
/// </summary>
public class MatrixGenerator
{
    private readonly double _min;
    private readonly double _max;
    private readonly Random _random = new();

    /// <summary>
    /// Initialize new MatrixGenerator that uses values in range <paramref name="min"/>..<paramref name="max"/>
    /// </summary>
    /// <param name="min">Min value for RNG</param>
    /// <param name="max">Max value for RNG (exclusive)</param>
    public MatrixGenerator(double min, double max)
    {
        _min = min;
        _max = max;
    }

    /// <summary>
    /// Generate new matrix
    /// </summary>
    /// <param name="rows">Number of rows</param>
    /// <param name="columns">Number of columns</param>
    /// <returns>Generated matrix</returns>
    public double[,] NewMatrix(int rows, int columns)
    {
        var result = new double[rows, columns];
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < columns; j++)
            {
                result[i, j] = NewValue();
            }
        }

        return result;
    }

    /// <summary>
    /// Generate pair of matrices that can be multiplied
    /// </summary>
    /// <param name="aRows">Number of rows of the first matrix</param>
    /// <param name="aColumns">Number of columns of the first matrix (= number of rows of the second matrix)</param>
    /// <param name="bColumns">Number of columns of the second matrix</param>
    /// <returns>Pair of matrices</returns>
    public (double[,] A, double[,] B) NewMatrixPair(int aRows, int aColumns, int bColumns)
        => (NewMatrix(aRows, aColumns), NewMatrix(aColumns, bColumns));

    /// <summary>
    /// Generate new value to use in matrix
    /// </summary>
    /// <returns>Random value in range</returns>
    private double NewValue() => _random.NextDouble() * (_max - _min) + _min;
}