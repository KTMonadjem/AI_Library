using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction;

public class LinearActivator : IActivationFunction
{
    public double Delta { get; set; }

    /// <summary>
    ///     y = x
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double Activate(double input)
    {
        Delta = Derive(input);
        return input;
    }

    /// <summary>
    ///     y' = 1
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public double Derive(double x)
    {
        return 1;
    }
}
