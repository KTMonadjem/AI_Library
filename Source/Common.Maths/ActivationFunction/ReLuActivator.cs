using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction;

public class ReLuActivator : IActivationFunction
{
    /// <summary>
    ///     y = x if x > 0
    ///     y = 0 if x < 0
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        return (Math.Max(input, 0), Derive(input));
    }

    /// <summary>
    ///     y = x
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs)
    {
        return (inputs.PointwiseMaximum(0.0), Derive(inputs));
    }

    public (Vector<double> Outputs, Vector<double> Derivatives) Activate1(Vector<double> inputs)
    {
        var outputs = new double[inputs.Count];
        var derivatives = new double[inputs.Count];

        for (var i = 0; i < inputs.Count; i++)
        {
            (outputs[i], derivatives[i]) = Activate(inputs[i]);
        }

        return (
            Vector<double>.Build.DenseOfArray(outputs),
            Vector<double>.Build.DenseOfArray(derivatives)
        );
    }

    /// <summary>
    ///     y' = 1 if x >= 0
    ///     y' = 0 if x < 0
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private static double Derive(double x)
    {
        return x >= 0 ? 1 : 0;
    }

    /// <summary>
    ///     y' = 1 if x >= 0
    ///     y' = 0 if x < 0
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    private static Vector<double> Derive(Vector<double> inputs)
    {
        return inputs
            .Add(0.000000000000001)
            .PointwiseMinimum(1.0)
            .PointwiseCeiling()
            .PointwiseMaximum(0.0);
    }
}
