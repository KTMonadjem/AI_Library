using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction;

public class ELuActivator : IActivationFunction
{
    private readonly double _alpha;

    /// <summary>
    ///     Create an ELu activator with Alpha
    /// </summary>
    /// <param name="alpha">A constant to multiply the exp by</param>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public ELuActivator(double alpha)
    {
        if (alpha < 0)
            throw new ArgumentOutOfRangeException(nameof(alpha));

        _alpha = alpha;
    }

    /// <summary>
    ///     y = x if x > 0
    ///     y = Alpha * (e^x - 1)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        var beta = input > 0 ? input : _alpha * (Math.Pow(Math.E, input) - 1);
        return (beta, Derive(input, beta));
    }

    // public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs)
    // {
    //     var zeroMaximums = inputs.PointwiseMaximum(0.0);
    //     var zeroMinimums = inputs.PointwiseMinimum(0.0);
    //
    //     var e =  Vector<double>.Build.DenseOfEnumerable(inputs.Select(_ => Math.E));
    //     var betas = zeroMaximums + _alpha * (e.PointwisePower(zeroMinimums) - 1);
    //
    //     var derivatives = zeroMaximums.PointwiseCeiling() + _alpha +
    //
    //     return (betas);
    // }

    /// <summary>
    ///     y' = 1 if x > 0
    ///     y' = Alpha * (e^x - 1) + Alpha
    /// </summary>
    /// <param name="x"></param>
    /// <param name="beta"></param>
    /// <returns></returns>
    private double Derive(double x, double beta)
    {
        return x >= 0 ? 1 : beta + _alpha;
    }
}
