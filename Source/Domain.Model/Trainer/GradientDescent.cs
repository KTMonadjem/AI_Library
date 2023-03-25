using ANN.Interface;
using Training.Algorithm.Interface;
using Training.LearningRate.Interface;
using Training.LossFunction.Interface;

namespace Learning.Supervised.ANN.Trainer
{
    public class GradientDescent: ITrainer
    {
        public ILearningRate LearningRate { get; set; }
        public ILossFunction LossFunction { get; set; }
        public ILearner Learner { get; set; }  

        public GradientDescent(ILearningRate learningRate, ILossFunction lossFunction, IANN ann)
        {
            LearningRate = learningRate;
            LossFunction = lossFunction;
            Learner = ann;
        }

        public void Train()
        {

        }
    }
}