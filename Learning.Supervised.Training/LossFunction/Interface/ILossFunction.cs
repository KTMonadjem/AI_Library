using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Training.LossFunction.Interface;

public interface ILossFunction
{
    public double CalculateLoss(Vector<double> expected, Vector<double> actual);
    public double CalculateLoss(double expected, double actual);
}
