using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction;

public class TanhActivator : TanhDerivative, IActivationFunction
{
    public double Delta { get; set; }

    /// <summary>
    ///     y = tanh(x)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double Activate(double input)
    {
        Tanh = Trig.Tanh(input);
        Delta = Derive(input);
        return Tanh;
    }
}