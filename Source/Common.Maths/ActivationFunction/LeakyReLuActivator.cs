using Common.Maths.ActivationFunction.Interface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Maths.ActivationFunction
{
    public class LeakyReLuActivator : IActivationFunction
    {
        private readonly double _leak;

        public LeakyReLuActivator(double leak) 
        {
            if (leak < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(leak));
            }

            _leak = leak;
        }

        public double Activate(double input)
        {
            return input > 0 ? input : _leak * input;
        }
    }
}
