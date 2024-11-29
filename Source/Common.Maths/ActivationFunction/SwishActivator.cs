using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction;

public class SwishActivator : SwishDerivative, IActivationFunction
{
    public double Delta { get; set; }

    /// <summary>
    ///     y = x * sigmoid(x)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double Activate(double input)
    {
        SigmoidX = SpecialFunctions.Logistic(input);
        SwishX = input * SigmoidX;
        Delta = Derive(input);
        return SwishX;
    }
}