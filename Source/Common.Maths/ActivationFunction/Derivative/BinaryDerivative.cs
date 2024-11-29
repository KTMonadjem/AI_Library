using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction.Derivative;

public class BinaryDerivative : IActivationDerivative
{
    /// <summary>
    ///     y' = 0
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public double Derive(double x)
    {
        return 0;
    }
}