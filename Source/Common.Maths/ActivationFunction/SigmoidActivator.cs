using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Maths.ActivationFunction
{
    public class SigmoidActivator: IActivationFunction
    {
        public double Activate(double input)
        {
            return SpecialFunctions.Logistic(input);
        }
    }
}
