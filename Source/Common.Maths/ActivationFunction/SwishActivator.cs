using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction;

public class SwishActivator : IActivationFunction
{
    private double _sigmoidX;
    private double _swishX;

    public double Delta { get; set; }

    /// <summary>
    ///     y = x * sigmoid(x)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double Activate(double input)
    {
        _sigmoidX = SpecialFunctions.Logistic(input);
        _swishX = input * _sigmoidX;
        Delta = Derive(input);
        return _swishX;
    }

    /// <summary>
    ///     y' = x * sigmoid(x) + sigmoid(x)(1 - x * sigmoid(x))
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private double Derive(double x)
    {
        return _swishX + _sigmoidX * (1 - _swishX);
    }
}
