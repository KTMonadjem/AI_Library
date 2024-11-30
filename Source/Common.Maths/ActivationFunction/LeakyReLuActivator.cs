using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction;

public class LeakyReLuActivator : IActivationFunction
{
    private readonly double _leak;

    /// <summary>
    ///     y = x if x > 0
    ///     y = Leak * x if x < 0
    /// </summary>
    /// <param name="leak"></param>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public LeakyReLuActivator(double leak)
    {
        if (leak < 0)
            throw new ArgumentOutOfRangeException(nameof(leak));

        _leak = leak;
    }

    public double Delta { get; set; }

    /// <summary>
    ///     A step function that returns a linear function for x > 0 and
    ///     a close to 0 linear function for x < 0
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double Activate(double input)
    {
        Delta = Derive(input);
        return input > 0 ? input : _leak * input;
    }

    /// <summary>
    ///     y' = 1 if x >= 0
    ///     y' = Leak if x < 0
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private double Derive(double x)
    {
        return x >= 0 ? 1 : _leak;
    }
}
