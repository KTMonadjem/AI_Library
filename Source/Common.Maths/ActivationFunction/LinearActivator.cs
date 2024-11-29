using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction;

public class LinearActivator : LinearDerivative, IActivationFunction
{
    public double Delta { get; set; }

    /// <summary>
    ///     y = x
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double Activate(double input)
    {
        Delta = Derive(input);
        return input;
    }
}