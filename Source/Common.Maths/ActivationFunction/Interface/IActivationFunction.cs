using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Maths.ActivationFunction.Interface
{
    public interface IActivationFunction
    {
        public abstract double Activate(double input);
    }
}
