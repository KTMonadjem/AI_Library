using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction;

public class SwishActivator : IActivationFunction
{
    private readonly double _beta;

    private SwishActivator(double beta)
    {
        _beta = beta;
    }

    public static SwishActivator Create(double b)
    {
        if (b < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(b));
        }

        return new SwishActivator(b);
    }

    /// <summary>
    ///     y = x * sigmoid(bx)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        var sigmoidBx = SpecialFunctions.Logistic(input * _beta);
        var swishX = input * sigmoidBx;
        return (swishX, Derive(swishX, sigmoidBx));
    }

    /// <summary>
    ///     y = x * sigmoid(bx)
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs)
    {
        var sigmoids = inputs.Multiply(-_beta).PointwiseExp().Add(1.0).PointwisePower(-1.0);
        var swishes = inputs.PointwiseMultiply(sigmoids);

        return (swishes, Derive(swishes, sigmoids));
    }

    /// <summary>
    ///     y' = b * swish(x) + sigmoid(x)(1 - b * sigmoid(x))
    /// </summary>
    /// <param name="swishX"></param>
    /// <param name="sigmoidBx"></param>
    /// <returns></returns>
    private double Derive(double swishX, double sigmoidBx)
    {
        var bSwishX = _beta * swishX;
        return bSwishX + sigmoidBx * (1 - bSwishX);
    }

    /// <summary>
    ///     y' = b * swish(x) + sigmoid(x)(1 - b * sigmoid(x))
    /// </summary>
    /// <param name="swishes"></param>
    /// <param name="sigmoids"></param>
    /// <returns></returns>
    private Vector<double> Derive(Vector<double> swishes, Vector<double> sigmoids)
    {
        var bSwishes = swishes.Multiply(_beta);

        return bSwishes.Add(sigmoids.PointwiseMultiply(bSwishes.Multiply(-1).Add(1)));
    }
}
