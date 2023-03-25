using Training.LossFunction.Interface;
using Training.LearningRate.Interface;

namespace Training.Algorithm.Interface
{
    public interface ITrainer
    {
        public ILearningRate LearningRate { get; set; }
        public ILossFunction LossFunction { get; set; }
        public ILearner Learner { get; set; }
        public void Train();
    }
}
