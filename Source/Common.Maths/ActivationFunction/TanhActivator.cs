using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

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
    ///     y' = 1 - tanh^2(x)
    /// </summary>
    /// <param name="tanh"></param>
    /// <returns></returns>
    private static double Derive(double tanh)
    {
        return 1 - Math.Pow(tanh, 2);
    }
}
