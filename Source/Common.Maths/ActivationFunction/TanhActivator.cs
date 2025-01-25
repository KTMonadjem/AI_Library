using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction;

public class TanhActivator : IActivationFunction
{
    /// <summary>
    ///     y = tanh(x)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        var tanh = Trig.Tanh(input);
        return (tanh, Derive(tanh));
    }

    /// <summary>
    ///     y = tanh(x)
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs)
    {
        var tanhs = inputs.PointwiseTanh();

        return (tanhs, Derive(tanhs));
    }

    /// <summary>
    ///     y' = 1 - tanh^2(x)
    /// </summary>
    /// <param name="tanh"></param>
    /// <returns></returns>
    private static double Derive(double tanh)
    {
        return 1 - Math.Pow(tanh, 2);
    }

    /// <summary>
    ///     y' = 1 - tanh^2(x)
    /// </summary>
    /// <param name="tanhs"></param>
    /// <returns></returns>
    private static Vector<double> Derive(Vector<double> tanhs)
    {
        return tanhs.PointwisePower(2).Multiply(-1.0).Add(1.0);
    }
}
