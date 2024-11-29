using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction.Derivative;

public class TanhDerivative : IActivationDerivative
{
    protected double Tanh;

    /// <summary>
    ///     y' = 1 - tanh^2(x)
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public double Derive(double x)
    {
        return 1 - Math.Pow(Tanh, 2);
    }
}