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
}
