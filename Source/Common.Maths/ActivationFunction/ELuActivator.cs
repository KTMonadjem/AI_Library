using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction;

public class ELuActivator : IActivationFunction
{
    private readonly double _alpha;

    private ELuActivator(double alpha)
    {
        if (alpha < 0)
            throw new ArgumentOutOfRangeException(nameof(alpha));

        _alpha = alpha;
    }

    /// <summary>
    ///     Create an ELu activator with Alpha
    /// </summary>
    /// <param name="alpha">A constant to multiply the exp by</param>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static ELuActivator Create(double alpha)
    {
        if (alpha < 0)
            throw new ArgumentOutOfRangeException(nameof(alpha));

        return new ELuActivator(alpha);
    }

    /// <summary>
    ///     y = x if x > 0
    ///     y = alpha * e^x - alpha if x <= 0
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        var outputs = input > 0 ? input : _alpha * (Math.Pow(Math.E, input) - 1);
        return (outputs, Derive(input, outputs));
    }

    /// <summary>
    ///     y = x if x > 0
    ///     y = alpha * e^x - alpha if x <= 0
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs)
    {
        var outputs = new double[inputs.Count];
        var derivatives = new double[inputs.Count];

        for (var i = 0; i < inputs.Count; i++)
        {
            (outputs[i], derivatives[i]) = Activate(inputs[i]);
        }

        return (
            Vector<double>.Build.DenseOfArray(outputs),
            Vector<double>.Build.DenseOfArray(derivatives)
        );
    }

    /// <summary>
    ///     y' = 1 if x > 0, otherwise
    ///     y' = (alpha * e^x - alpha) + alpha
    ///     y' = output + alpha
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <returns></returns>
    private double Derive(double input, double output)
    {
        return input >= 0 ? 1 : output + _alpha;
    }
}
