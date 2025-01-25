namespace Learning.Supervised.Training.LearningRate.Interface;

public interface ILearningRate
{
    public double Apply(double errorSignal);
}
