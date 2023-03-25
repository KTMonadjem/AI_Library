using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Training.LearningRate.Interface;

namespace Training.LearningRate
{
    public class FlatLearningRate: ILearningRate
    {
        private readonly double _learningRate;
        public FlatLearningRate(double learningRate)
        {
            _learningRate = learningRate;
        }

        public double ApplyLearningRate(double errorSignal)
        {
            return _learningRate * errorSignal;
        }
    }
}
