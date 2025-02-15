using BenchmarkDotNet.Running;

namespace matrix_multiplication_cpu_optimization;

public static class Program
{
    static void Main(string[] args)
    {
        BenchmarkRunner.Run<MatrixMultiplicationCpuOptimizationBenchmark>();
    }
}