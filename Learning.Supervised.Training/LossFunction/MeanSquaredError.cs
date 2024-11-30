using Learning.Supervised.Training.LossFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Training.LossFunction;

public class MeanSquaredError : ILossFunction
{
    public Vector<double> CalculateLoss(Vector<double> expected, Vector<double> actual)
    {
        if (actual.Count != expected.Count)
            throw new ArgumentException("Expected and actual should be the same length.");

        return expected.Subtract(actual).PointwisePower(2).PointwiseSqrt();
    }
}
