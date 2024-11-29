using Learning.Supervised.Training.LearningRate.Interface;
using Learning.Supervised.Training.LossFunction.Interface;

namespace Learning.Supervised.Training.Algorithm.Interface;

public interface ITrainer
{
    public ILearningRate LearningRate { get; set; }
    public ILossFunction LossFunction { get; set; }
    public ILearner Learner { get; set; }
    public void Train();
}