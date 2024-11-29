using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction;

public class SigmoidActivator : SigmoidDerivative, IActivationFunction
{
    public double Delta { get; set; }

    /// <summary>
    ///     y = sigmoid(x)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double Activate(double input)
    {
        SigmoidX = SpecialFunctions.Logistic(input);
        Delta = Derive(input);
        return SigmoidX;
    }
}