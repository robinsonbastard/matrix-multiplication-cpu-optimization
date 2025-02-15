namespace matrix_multiplication_cpu_optimization;

/// <summary>
/// Solver that can multiply matrices
/// </summary>
public interface ISolver
{
    /// <summary>
    /// Multiply two matrices (if they can be multiplied)
    /// </summary>
    /// <param name="a">First matrix (NxM)</param>
    /// <param name="b">Second matrix (MxK)</param>
    /// <returns>Result of the multiplication</returns>
    double[,] Multiply(double[,] a, double[,] b);
}