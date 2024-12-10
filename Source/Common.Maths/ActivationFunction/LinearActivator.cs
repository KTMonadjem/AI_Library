using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction;

public class LinearActivator : IActivationFunction
{
    /// <summary>
    ///     y = x
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        return (input, Derive(input));
    }

    public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs)
    {
        return (
            inputs.PointwiseMaximum(0.0),
            Vector<double>.Build.DenseOfEnumerable(inputs.Select(_ => 1.0))
        );
    }

    /// <summary>
    ///     y' = 1
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private static double Derive(double x)
    {
        return 1;
    }
}
