using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction;

public class LeakyReLuActivator : IActivationFunction
{
    private readonly double _leak;

    private LeakyReLuActivator(double leak)
    {
        _leak = leak;
    }

    public static LeakyReLuActivator Create(double leak)
    {
        if (leak < 0)
            throw new ArgumentOutOfRangeException(nameof(leak));

        return new LeakyReLuActivator(leak);
    }

    /// <summary>
    ///     y = x if x >= 0
    ///     y = Leak * x if x < 0
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        return (input > 0 ? input : _leak * input, Derive(input));
    }

    /// <summary>
    ///     y = x if x >= 0
    ///     y = Leak * x if x < 0
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs)
    {
        var outputs = inputs
            .PointwiseMaximum(0.0)
            .Add(inputs.PointwiseMinimum(0.0).Multiply(_leak));

        return (outputs, Derive(inputs));
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
    ///     y' = Leak if x < 0
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private double Derive(double x)
    {
        return x >= 0 ? 1 : _leak;
    }

    /// <summary>
    ///     y' = 1 if x >= 0
    ///     y' = Leak if x < 0
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    private Vector<double> Derive(Vector<double> inputs)
    {
        return inputs
            .Add(0.000000000000001)
            .PointwiseMinimum(1.0)
            .PointwiseCeiling()
            .PointwiseMaximum(_leak);
    }
}
