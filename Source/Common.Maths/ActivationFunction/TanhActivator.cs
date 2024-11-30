using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction;

public class TanhActivator : IActivationFunction
{
    private double _tanh;
    public double Delta { get; set; }

    /// <summary>
    ///     y = tanh(x)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double Activate(double input)
    {
        _tanh = Trig.Tanh(input);
        Delta = Derive(input);
        return _tanh;
    }

    /// <summary>
    ///     y' = 1 - tanh^2(x)
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private double Derive(double x)
    {
        return 1 - Math.Pow(_tanh, 2);
    }
}
