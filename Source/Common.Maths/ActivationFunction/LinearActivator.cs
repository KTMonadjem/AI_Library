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
        return (input, 1.0);
    }

    /// <summary>
    ///     y = x
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs)
    {
        return (inputs, Vector<double>.Build.DenseOfEnumerable(inputs.Select(_ => 1.0)));
    }
}
