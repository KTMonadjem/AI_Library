using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Maths.ActivationFunction
{
    public class ReLuActivator : ReLuDerivative, IActivationFunction
    {
        public double Delta { get; set; }

        /// <summary>
        /// y = x if x > 0 
        /// y = 0 if x < 0
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Activate(double input)
        {
            Delta = Derive(input);
            return Math.Max(input, 0);
        }
    }
}
