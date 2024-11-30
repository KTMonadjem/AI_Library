using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction;

public class SigmoidActivator : IActivationFunction
{
    private double _sigmoidX;

    public double Delta { get; set; }

    /// <summary>
    ///     y = sigmoid(x)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double Activate(double input)
    {
        _sigmoidX = SpecialFunctions.Logistic(input);
        Delta = Derive(input);
        return _sigmoidX;
    }

    /// <summary>
    ///     y' = sigmoid(x) * (1 - sigmoid(x))
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private double Derive(double x)
    {
        var log = _sigmoidX;
        return log * (1 - log);
    }
}
