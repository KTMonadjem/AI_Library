using ANN.Interface;
using Training.Algorithm.Interface;
using Training.LearningRate.Interface;
using Training.LossFunction.Interface;

namespace Learning.Supervised.ANN.Trainer
{
    public class GradientDescent: ITrainer
    {
        public required ILearningRate LearningRate { get; set; }
        public required ILossFunction LossFunction { get; set; }
        public required ILearner Learner { get; set; }  

        public GradientDescent(ILearningRate learningRate, ILossFunction lossFunction, IANN ann)
        {
            LearningRate = learningRate;
            LossFunction = lossFunction;
            Learner = ann;
        }

        public void Train()
        {
            if (Learner.GetType() == typeof(ANN))
            {
                TrainANN();
            }
        }

        private void TrainANN()
        {
            try
            {
                var ann = Learner as ANN;
                if (ann is null)
                {
                    throw new NullReferenceException("Cannot train null ANN.");
                }

                if (!ann.HasBeenBuilt)
                {
                    ann.Build();
                }

                // This will run the ann if it hasn't already been
                var outputs = ann.Outputs;
            }
            catch (Exception e)
            {
                throw new Exception("Error running the ANN before gradient descent: ", e);
            }
        }
    }
}