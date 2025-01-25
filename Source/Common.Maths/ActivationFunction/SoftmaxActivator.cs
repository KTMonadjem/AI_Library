using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction;

public class SoftmaxActivator : IActivationFunction
{
    /// <summary>
    ///     y = sigmoid(x)
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public (double Output, double Derivative) Activate(double input)
    {
        throw new NotImplementedException();
    }

    public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    ///     y' = sigmoid(x) * (1 - sigmoid(x))
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private double Derive(double x)
    {
        throw new NotImplementedException();
    }
}
