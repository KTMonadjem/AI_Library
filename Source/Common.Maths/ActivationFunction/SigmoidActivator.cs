using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction;

public class SigmoidActivator : IActivationFunction
{
    /// <summary>
    ///     y = sigmoid(x)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        var sigmoidX = SpecialFunctions.Logistic(input);
        return (sigmoidX, Derive(sigmoidX));
    }

    /// <summary>
    ///     y' = sigmoid(x) * (1 - sigmoid(x))
    /// </summary>
    /// <param name="sigmoidX"></param>
    /// <returns></returns>
    private static double Derive(double sigmoidX)
    {
        return sigmoidX * (1 - sigmoidX);
    }
}
