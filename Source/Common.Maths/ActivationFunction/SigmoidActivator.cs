using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction;

public class SigmoidActivator : IActivationFunction
{
    /// <summary>
    ///     y = sigmoid(x)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        var sigmoidX = SpecialFunctions.Logistic(input);
        return (sigmoidX, Derive(sigmoidX));
    }

    /// <summary>
    ///     y = sigmoid(x)
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs)
    {
        var sigmoids = inputs.Multiply(-1.0).PointwiseExp().Add(1.0).PointwisePower(-1.0);

        return (sigmoids, Derive(sigmoids));
    }

    /// <summary>
    ///     y' = sigmoid(x) * (1 - sigmoid(x))
    /// </summary>
    /// <param name="sigmoidX"></param>
    /// <returns></returns>
    private static double Derive(double sigmoidX)
    {
        return sigmoidX * (1 - sigmoidX);
    }

    /// <summary>
    ///     y' = sigmoid(x) * (1 - sigmoid(x))
    /// </summary>
    /// <param name="sigmoids"></param>
    /// <returns></returns>
    private static Vector<double> Derive(Vector<double> sigmoids)
    {
        return sigmoids.PointwiseMultiply(sigmoids.Multiply(-1.0).Add(1.0));
    }
}
