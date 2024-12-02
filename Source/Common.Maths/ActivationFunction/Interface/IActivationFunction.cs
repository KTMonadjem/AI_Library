using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction.Interface;

public interface IActivationFunction
{
    public enum ActivationFunction
    {
        Binary,
        Linear,
        ReLu,
        LeakyReLu,
        ELu,
        Sigmoid,
        Tanh,
        Swish,
        Softmax,
    }

    public (double Output, double Derivative) Activate(double input);

    // TODO: Implement vector activations
    // public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs);
}
