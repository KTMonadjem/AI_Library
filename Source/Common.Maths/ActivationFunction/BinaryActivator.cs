using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction;

public class BinaryActivator : IActivationFunction
{
    public double Delta { get; set; }

    /// <summary>
    ///     y = 1 if x > 0
    ///     y = 0 if x <= 0
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double Activate(double input)
    {
        Delta = Derive(input);
        return input > 0 ? 1 : 0;
    }

    /// <summary>
    ///     y' = 0
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private static double Derive(double x)
    {
        return 0;
    }
}
