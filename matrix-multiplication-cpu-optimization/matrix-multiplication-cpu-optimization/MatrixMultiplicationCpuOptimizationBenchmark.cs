using BenchmarkDotNet.Attributes;

namespace matrix_multiplication_cpu_optimization;

public class MatrixMultiplicationCpuOptimizationBenchmark
{
    private readonly NaiveSolver _naiveSolver = new();
    private readonly StrassenSolver _strassenSolver = new();
    private readonly MatrixGenerator _generator = new(-100, 100);
    
    [Params(10, 20, 40)]
    public int MatrixSize;

    [Benchmark]
    public void Naive()
    {
        var matrix = _generator.NewMatrixPair(MatrixSize, MatrixSize, MatrixSize);
        _naiveSolver.Multiply(matrix.Item1, matrix.Item2);
    }
    
    [Benchmark]
    public void Strassen()
    {
        var matrix = _generator.NewMatrixPair(MatrixSize, MatrixSize, MatrixSize);
        _strassenSolver.Multiply(matrix.Item1, matrix.Item2);
    }
}