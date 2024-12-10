using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction;

public class SwishActivator : IActivationFunction
{
    public double Delta { get; private set; }

    /// <summary>
    ///     y = x * sigmoid(x)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        var sigmoidX = SpecialFunctions.Logistic(input);
        var swishX = input * sigmoidX;
        return (swishX, Derive(swishX, sigmoidX));
    }

    /// <summary>
    ///     y' = x * sigmoid(x) + sigmoid(x)(1 - x * sigmoid(x))
    /// </summary>
    /// <param name="swishX"></param>
    /// <param name="sigmoidX"></param>
    /// <returns></returns>
    private static double Derive(double swishX, double sigmoidX)
    {
        return swishX + sigmoidX * (1 - swishX);
    }
}
