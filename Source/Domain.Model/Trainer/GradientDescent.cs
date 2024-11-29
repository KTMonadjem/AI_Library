using ANN.Interface;
using Data.DataSet.Interface;
using MathNet.Numerics.LinearAlgebra;
using Learning.Supervised.Training.Algorithm.Interface;
using Learning.Supervised.Training.LearningRate.Interface;
using Learning.Supervised.Training.LossFunction.Interface;

namespace Learning.Supervised.ANN.Trainer
{
    public class GradientDescent: ITrainer
    {
        public required ILearningRate LearningRate { get; set; }
        public required ILossFunction LossFunction { get; set; }
        public required ILearner Learner { get; set; }  
        public required IDataSet DataSet { get; set; }

        public GradientDescent(ILearningRate learningRate, ILossFunction lossFunction, IANN ann, IDataSet dataSet)
        {
            LearningRate = learningRate;
            LossFunction = lossFunction;
            Learner = ann;
            DataSet = dataSet;
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
            var ann = Learner as ANN;
            if (ann is null)
            {
                throw new NullReferenceException("Cannot train null ANN.");
            }
            Vector<double> outputs;

            try
            {
                if (!ann.HasBeenBuilt)
                {
                    ann.Build();
                }

                // This will run the ann if it hasn't already been
                outputs = ann.Outputs;
            }
            catch (Exception e)
            {
                throw new Exception("Error running the ANN during gradient descent: ", e);
            }


        }
    }
}