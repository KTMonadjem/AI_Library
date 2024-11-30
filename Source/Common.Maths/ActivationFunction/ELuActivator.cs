using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction;

public class ELuActivator : IActivationFunction
{
    private readonly double _alpha;
    private double _beta;

    /// <summary>
    ///     Create an ELu activator with Alpha
    /// </summary>
    /// <param name="alpha">A constant to multiply the exp by</param>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public ELuActivator(double alpha)
    {
        if (alpha < 0)
            throw new ArgumentOutOfRangeException(nameof(alpha));

        _alpha = alpha;
    }

    public double Delta { get; set; }

    /// <summary>
    ///     y = x if x > 0
    ///     y = Alpha * (e^x - 1)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double Activate(double input)
    {
        _beta = input > 0 ? input : _alpha * (Math.Pow(Math.E, input) - 1);
        Delta = Derive(input);
        return _beta;
    }

    /// <summary>
    ///     y' = 1 if x > 0
    ///     y' = Alpha * (e^x - 1) + Alpha
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private double Derive(double x)
    {
        return x >= 0 ? 1 : _beta + _alpha;
    }
}
