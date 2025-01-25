using Learning.Supervised.Training.LossFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Training.LossFunction;

// TODO: Add vector operations

public class MeanSquaredError : ILossFunction
{
    public double CalculateLoss(Vector<double> expected, Vector<double> actual)
    {
        if (actual.Count != expected.Count)
            throw new ArgumentException("Expected and actual should be the same length.");

        var sum = actual.Select((t, i) => CalculateLoss(expected[i], t)).Sum();
        return sum / actual.Count;
    }

    public double CalculateLoss(double expected, double actual)
    {
        return Math.Pow(actual - expected, 2);
    }
}
