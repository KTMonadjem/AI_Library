using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Maths.ActivationFunction
{
    public class LeakyReLuActivator : LeakyReLuDerivative, IActivationFunction
    {
        private readonly double _leak;

        public double Delta { get; set; }

        /// <summary>
        /// y = x if x > 0
        /// y = Leak * x if x < 0
        /// </summary>
        /// <param name="leak"></param>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public LeakyReLuActivator(double leak) : base(leak)
        {
            if (leak < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(leak));
            }

            _leak = leak;
        }

        /// <summary>
        /// A step function that returns a linear function for x > 0 and 
        /// a close to 0 linear function for x < 0
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Activate(double input)
        {
            Delta = Derive(input);
            return input > 0 ? input : _leak * input;
        }
    }
}
