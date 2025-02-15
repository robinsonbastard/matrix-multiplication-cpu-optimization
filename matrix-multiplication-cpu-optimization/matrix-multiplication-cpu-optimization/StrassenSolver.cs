namespace matrix_multiplication_cpu_optimization;

/// <summary>
/// Solver that multiplies matrices using Strassen algorithm
/// </summary>
public class StrassenSolver : ISolver
{
    private const int FallbackDimension = 64;
    private readonly NaiveSolver _naiveSolver = new NaiveSolver();

    /// <inheritdoc/>
    public double[,] Multiply(double[,] a, double[,] b)
    {
        var (rows, cols) = (a.GetLength(0), b.GetLength(1));

        // Get highest dimension of both matrices
        var dimension = Math.Max(
            Math.Max(a.GetLength(0), a.GetLength(1)),
            Math.Max(b.GetLength(0), b.GetLength(1))
        );

        dimension = CalculateDesiredDimension(dimension);

        // Use naive solver if matrices are too small
        if (dimension <= FallbackDimension)
        {
            return _naiveSolver.Multiply(a, b);
        }

        // Resize A if necessary
        if (a.GetLength(0) != dimension || a.GetLength(1) != dimension)
        {
            a = Extend(a, dimension);
        }

        // Resize B if necessary
        if (b.GetLength(0) != dimension || b.GetLength(1) != dimension)
        {
            b = Extend(b, dimension);
        }

        var result = MultiplyRecursive(a, b);

        // Resize resulting matrix back to correct dimensions
        return Shrink(result, rows, cols, 0, 0);
    }

    /// <summary>
    /// Multiply matrices recursively
    /// </summary>
    /// <param name="a">First matrix</param>
    /// <param name="b">Second matrix</param>
    /// <returns>The result of multiplication</returns>
    private double[,] MultiplyRecursive(double[,] a, double[,] b)
    {
        var dimension = a.GetLength(0);

        // Use naive solver if matrices are too small
        if (dimension <= FallbackDimension)
        {
            return _naiveSolver.Multiply(a, b);
        }

        dimension = CalculateDesiredDimension(dimension);

        var (a11, a12, a21, a22) = SubdivideSquareMatrix(a, dimension);
        var (b11, b12, b21, b22) = SubdivideSquareMatrix(b, dimension);

        var m1 = MultiplyRecursive(a12.Subtract(a22), b21.Add(b22));
        var m2 = MultiplyRecursive(a11.Add(a22), b11.Add(b22));
        var m3 = MultiplyRecursive(a11.Subtract(a21), b11.Add(b12));
        var m4 = MultiplyRecursive(a11.Add(a12), b22);
        var m5 = MultiplyRecursive(a11, b12.Subtract(b22));
        var m6 = MultiplyRecursive(a22, b21.Subtract(b11));
        var m7 = MultiplyRecursive(a21.Add(a22), b11);

        // Result
        // c11 c12
        // c21 c22
        var c = new double[dimension, dimension];
        var halfDimension = dimension / 2;

        // Fill resulting matrix
        for (var i = 0; i < halfDimension; i++)
        {
            for (var j = 0; j < halfDimension; j++)
            {
                // c11
                c[i, j] = m1[i, j] + m2[i, j] - m4[i, j] + m6[i, j];
                // c12
                c[i, halfDimension + j] = m4[i, j] + m5[i, j];
                // c21
                c[halfDimension + i, j] = m6[i, j] + m7[i, j];
                // c22
                c[halfDimension + i, halfDimension + j] = m2[i, j] - m3[i, j] + m5[i, j] - m7[i, j];
            }
        }

        return c;
    }

    /// <summary>
    /// Append extra rows and columns to matrix to make it square with given dimension
    /// </summary>
    /// <param name="x">Matrix to append to</param>
    /// <param name="dimension">Dimension to extend to</param>
    /// <returns>Square matrix with given <paramref name="dimension"/></returns>
    private static double[,] Extend(double[,] x, int dimension)
    {
        var result = new double[dimension, dimension];
        var (rows, columns) = (x.GetLength(0), x.GetLength(1));

        // Iterate over rows of the matrix
        for (var i = 0; i < rows; i++)
        {
            // Copy values
            for (var j = 0; j < columns; j++)
            {
                result[i, j] = x[i, j];
            }

            // Fill extra columns with zeros
            for (var j = columns; j < dimension; j++)
            {
                result[i, j] = 0d;
            }
        }

        // Fill extra rows with zeros
        for (var i = rows; i < dimension; i++)
        {
            for (var j = 0; j < dimension; j++)
            {
                result[i, j] = 0d;
            }
        }

        return result;
    }

    /// <summary>
    /// Remove extra rows and columns of the matrix
    /// </summary>
    /// <param name="x">Matrix to shrink</param>
    /// <param name="rows">Rows that should be left</param>
    /// <param name="columns">Columns that should be left</param>
    /// <param name="rowsOffset">Top rows to skip</param>
    /// <param name="columnsOffset">Left columns to skip</param>
    /// <returns>Matrix with given number of <paramref name="rows"/> and <paramref name="columns"/></returns>
    private static double[,] Shrink(double[,] x, int rows, int columns, int rowsOffset, int columnsOffset)
    {
        var result = new double[rows, columns];
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < columns; j++)
            {
                result[i, j] = x[rowsOffset + i, columnsOffset + j];
            }
        }

        return result;
    }

    /// <summary>
    /// Calculate desired dimension for given matrix dimension
    /// </summary>
    ///
    /// <para>Desired dimension:</para>
    /// <para>1. is as small as possible</para>
    /// <para>2. is even</para>
    /// <para>If <paramref name="dimension"/> is even, then desired dimension equals to <paramref name="dimension"/></para>
    /// <para>If <paramref name="dimension"/> is odd, desired dimension is <paramref name="dimension"/> + 1</para>
    ///
    /// <param name="dimension"></param>
    /// <returns>Desired dimension</returns>
    private static int CalculateDesiredDimension(int dimension) => dimension % 2 == 0 ? dimension : dimension + 1;

    /// <summary>
    /// Subdivide square matrix into 4 submatrices (append extra row and column if necessary)
    /// </summary>
    ///
    /// <para>For matrix X subdivision will work like that:</para>
    /// <para>x11 x12</para>
    /// <para>x21 x22</para>
    ///
    /// <param name="x">Square matrix</param>
    /// <param name="dimension">Desired dimension</param>
    /// <returns>Four submatrices</returns>
    private static (double[,], double[,], double[,], double[,]) SubdivideSquareMatrix(double[,] x, int dimension)
    {
        // Dimension of each submatrix
        var halfDimension = dimension / 2;

        var x11 = new double[halfDimension, halfDimension];
        var x12 = new double[halfDimension, halfDimension];
        var x21 = new double[halfDimension, halfDimension];
        var x22 = new double[halfDimension, halfDimension];

        // Real dimension of matrix X
        var xDimension = x.GetLength(0);

        for (var i = 0; i < halfDimension; i++)
        {
            // Fill leftmost upper submatrix
            for (var j = 0; j < halfDimension; j++)
            {
                x11[i, j] = x[i, j];
            }

            // Fill rightmost upper submatrix
            for (var j = 0; j < xDimension - halfDimension; j++)
            {
                x12[i, j] = x[i, halfDimension + j];
            }
        }

        for (var i = 0; i < xDimension - halfDimension; i++)
        {
            // Fill leftmost bottom submatrix
            for (var j = 0; j < halfDimension; j++)
            {
                x21[i, j] = x[halfDimension + i, j];
            }

            // Fill rightmost bottom submatrix
            for (var j = 0; j < xDimension - halfDimension; j++)
            {
                x22[i, j] = x[halfDimension + i, halfDimension + j];
            }
        }

        return (x11, x12, x21, x22);
    }
}