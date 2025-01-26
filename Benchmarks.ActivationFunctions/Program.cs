using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using Common.Maths.ActivationFunction;
using MathNet.Numerics.LinearAlgebra;

namespace Benchmarks.ActivationFunctions;

[CsvExporter]
[HtmlExporter]
[MemoryDiagnoser]
[ThreadingDiagnoser]
public class ReLuActivatorBenchmarks
{
    private static readonly VectorBuilder<double> V = Vector<double>.Build;
    private static readonly Vector<double> Data = V.DenseOfEnumerable(
        Enumerable.Range(-10000, 10000).Select(i => (double)i)
    );
    private static readonly ReLuActivator ReLuActivator = new();

    [Benchmark]
    public (Vector<double>, Vector<double>) MultipleReLuActivator()
    {
        var outputs = new double[20000];
        var derivatives = new double[20000];
        for (var i = 0; i < Data.Count; i++)
        {
            (outputs[i], derivatives[i]) = ReLuActivator.Activate(Data[i]);
        }
        return (V.DenseOfEnumerable(outputs), V.DenseOfEnumerable(derivatives));
    }

    [Benchmark]
    public (Vector<double>, Vector<double>) VectorReLuActivator()
    {
        return ReLuActivator.Activate(Data);
    }
}

[CsvExporter]
[HtmlExporter]
[MemoryDiagnoser]
[ThreadingDiagnoser]
public class LeakyReLuActivatorBenchmarks
{
    private static readonly VectorBuilder<double> V = Vector<double>.Build;
    private static readonly Vector<double> Data = V.DenseOfEnumerable(
        Enumerable.Range(-10000, 10000).Select(i => (double)i)
    );
    private static readonly LeakyReLuActivator LeakyReLuActivator = LeakyReLuActivator.Create(0.5);

    [Benchmark]
    public (Vector<double>, Vector<double>) MultipleLeakyReLuActivator()
    {
        var outputs = new double[20000];
        var derivatives = new double[20000];
        for (var i = 0; i < Data.Count; i++)
        {
            (outputs[i], derivatives[i]) = LeakyReLuActivator.Activate(Data[i]);
        }
        return (V.DenseOfEnumerable(outputs), V.DenseOfEnumerable(derivatives));
    }

    [Benchmark]
    public (Vector<double>, Vector<double>) VectorLeakyReLuActivator()
    {
        return LeakyReLuActivator.Activate(Data);
    }
}

[CsvExporter]
[HtmlExporter]
[MemoryDiagnoser]
[ThreadingDiagnoser]
public class BinaryActivatorBenchmarks
{
    private static readonly VectorBuilder<double> V = Vector<double>.Build;
    private static readonly Vector<double> Data = V.DenseOfEnumerable(
        Enumerable.Range(-10000, 10000).Select(i => (double)i)
    );
    private static readonly BinaryActivator BinaryActivator = new();

    [Benchmark]
    public (Vector<double>, Vector<double>) MultipleBinaryActivator()
    {
        var outputs = new double[20000];
        var derivatives = new double[20000];
        for (var i = 0; i < Data.Count; i++)
        {
            (outputs[i], derivatives[i]) = BinaryActivator.Activate(Data[i]);
        }
        return (V.DenseOfEnumerable(outputs), V.DenseOfEnumerable(derivatives));
    }

    [Benchmark]
    public (Vector<double>, Vector<double>) VectorBinaryActivator()
    {
        return BinaryActivator.Activate(Data);
    }
}

[CsvExporter]
[HtmlExporter]
[MemoryDiagnoser]
[ThreadingDiagnoser]
public class ELuActivatorBenchmarks
{
    private static readonly VectorBuilder<double> V = Vector<double>.Build;
    private static readonly Vector<double> Data = V.DenseOfEnumerable(
        Enumerable.Range(-10000, 10000).Select(i => (double)i)
    );
    private static readonly ELuActivator ELuActivator = ELuActivator.Create(0.5);

    [Benchmark]
    public (Vector<double>, Vector<double>) MultipleELuActivator()
    {
        var outputs = new double[20000];
        var derivatives = new double[20000];
        for (var i = 0; i < Data.Count; i++)
        {
            (outputs[i], derivatives[i]) = ELuActivator.Activate(Data[i]);
        }
        return (V.DenseOfEnumerable(outputs), V.DenseOfEnumerable(derivatives));
    }

    [Benchmark]
    public (Vector<double>, Vector<double>) VectorELuActivator()
    {
        return ELuActivator.Activate(Data);
    }
}

[CsvExporter]
[HtmlExporter]
[MemoryDiagnoser]
[ThreadingDiagnoser]
public class LinearActivatorBenchmarks
{
    private static readonly VectorBuilder<double> V = Vector<double>.Build;
    private static readonly Vector<double> Data = V.DenseOfEnumerable(
        Enumerable.Range(-10000, 10000).Select(i => (double)i)
    );
    private static readonly LinearActivator LinearActivator = new();

    [Benchmark]
    public (Vector<double>, Vector<double>) MultipleLinearActivator()
    {
        var outputs = new double[20000];
        var derivatives = new double[20000];
        for (var i = 0; i < Data.Count; i++)
        {
            (outputs[i], derivatives[i]) = LinearActivator.Activate(Data[i]);
        }
        return (V.DenseOfEnumerable(outputs), V.DenseOfEnumerable(derivatives));
    }

    [Benchmark]
    public (Vector<double>, Vector<double>) VectorLinearActivator()
    {
        return LinearActivator.Activate(Data);
    }
}

[CsvExporter]
[HtmlExporter]
[MemoryDiagnoser]
[ThreadingDiagnoser]
public class SigmoidActivatorBenchmarks
{
    private static readonly VectorBuilder<double> V = Vector<double>.Build;
    private static readonly Vector<double> Data = V.DenseOfEnumerable(
        Enumerable.Range(-10000, 10000).Select(i => (double)i)
    );
    private static readonly SigmoidActivator SigmoidActivator = new();

    [Benchmark]
    public (Vector<double>, Vector<double>) MultipleSigmoidActivator()
    {
        var outputs = new double[20000];
        var derivatives = new double[20000];
        for (var i = 0; i < Data.Count; i++)
        {
            (outputs[i], derivatives[i]) = SigmoidActivator.Activate(Data[i]);
        }
        return (V.DenseOfEnumerable(outputs), V.DenseOfEnumerable(derivatives));
    }

    [Benchmark]
    public (Vector<double>, Vector<double>) VectorSigmoidActivator()
    {
        return SigmoidActivator.Activate(Data);
    }
}

[CsvExporter]
[HtmlExporter]
[MemoryDiagnoser]
[ThreadingDiagnoser]
public class TanhActivatorBenchmarks
{
    private static readonly VectorBuilder<double> V = Vector<double>.Build;
    private static readonly Vector<double> Data = V.DenseOfEnumerable(
        Enumerable.Range(-10000, 10000).Select(i => (double)i)
    );
    private static readonly TanhActivator TanhActivator = new();

    [Benchmark]
    public (Vector<double>, Vector<double>) MultipleTanhActivator()
    {
        var outputs = new double[20000];
        var derivatives = new double[20000];
        for (var i = 0; i < Data.Count; i++)
        {
            (outputs[i], derivatives[i]) = TanhActivator.Activate(Data[i]);
        }
        return (V.DenseOfEnumerable(outputs), V.DenseOfEnumerable(derivatives));
    }

    [Benchmark]
    public (Vector<double>, Vector<double>) VectorTanhActivator()
    {
        return TanhActivator.Activate(Data);
    }
}

[CsvExporter]
[HtmlExporter]
[MemoryDiagnoser]
[ThreadingDiagnoser]
public class SwishActivatorBenchmarks
{
    private static readonly VectorBuilder<double> V = Vector<double>.Build;
    private static readonly Vector<double> Data = V.DenseOfEnumerable(
        Enumerable.Range(-10000, 10000).Select(i => (double)i)
    );
    private static readonly SwishActivator SwishActivator = SwishActivator.Create(0.5);

    [Benchmark]
    public (Vector<double>, Vector<double>) MultipleSwishActivator()
    {
        var outputs = new double[20000];
        var derivatives = new double[20000];
        for (var i = 0; i < Data.Count; i++)
        {
            (outputs[i], derivatives[i]) = SwishActivator.Activate(Data[i]);
        }
        return (V.DenseOfEnumerable(outputs), V.DenseOfEnumerable(derivatives));
    }

    [Benchmark]
    public (Vector<double>, Vector<double>) VectorSwishActivator()
    {
        return SwishActivator.Activate(Data);
    }
}

public class Program
{
    public static void Main(string[] args)
    {
        var summary = BenchmarkRunner.Run(
            typeof(Program).Assembly,
            new DebugBuildConfig().WithOptions(ConfigOptions.DisableOptimizationsValidator)
        );
    }
}
