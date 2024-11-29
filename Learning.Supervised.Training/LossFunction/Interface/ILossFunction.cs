namespace Learning.Supervised.Training.LossFunction.Interface;

public interface ILossFunction
{
    public double CalculateLoss(double[] expected, double[] actual);
    public double CalculateLoss(double expected, double actual);
}