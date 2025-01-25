using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction;

public class BinaryActivator : IActivationFunction
{
    /// <summary>
    ///     y = 1 if x > 0
    ///     y = 0 if x <= 0
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        return (input > 0 ? 1.0 : 0.0, 0.0);
    }

    /// <summary>
    /// Activate a vector of inputs
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs)
    {
        return (
            inputs.PointwiseMinimum(1.0).PointwiseCeiling().PointwiseMaximum(0.0),
            Vector<double>.Build.DenseOfEnumerable(inputs.Select(_ => 0.0))
        );
    }
}
