using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Maths.ActivationFunction
{
    public class LinearActivator: LinearDerivative, IActivationFunction
    {
        /// <summary>
        /// y = x
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Activate(double input)
        {
            return input;
        }
    }
}
