using Common.Maths.ActivationFunction.Interface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Maths.ActivationFunction
{
    public class ELuActivator: IActivationFunction
    {
        private readonly double _alpha;

        public ELuActivator(double alpha)
        {
            if (alpha < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(alpha));
            }

            _alpha = alpha;
        }

        public double Activate(double input)
        {
            return input > 0 ? input : _alpha * (Math.Pow(Math.E, input) - 1);
        }
    }
}
