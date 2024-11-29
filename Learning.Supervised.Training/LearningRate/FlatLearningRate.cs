using Learning.Supervised.Training.LearningRate.Interface;

namespace Learning.Supervised.Training.LearningRate;

public class FlatLearningRate : ILearningRate
{
    private readonly double _alpha;

    public FlatLearningRate(double alpha)
    {
        _alpha = alpha;
    }

    public double ApplyLearningRate(double errorSignal)
    {
        return _alpha * errorSignal;
    }
}