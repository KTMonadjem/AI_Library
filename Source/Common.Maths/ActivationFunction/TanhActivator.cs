using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Maths.ActivationFunction
{
    public class TanhActivator : IActivationFunction
    {
        public double Activate(double input)
        {
            return Trig.Tanh(input);
        }
    }
}
