using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Training.Algorithm.Interface;

namespace ANN.Interface
{
    public interface ISupervisedLearner
    {
        public ITrainer Trainer { get; }

        public void Run();
    }
}
