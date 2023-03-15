using Common.Maths.ActivationFunction.Interface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Maths.ActivationFunction
{
    public class ReLuActivator : IActivationFunction
    {
        public double Activate(double input)
        {
            return Math.Max(input, 0);
        }
    }
}
