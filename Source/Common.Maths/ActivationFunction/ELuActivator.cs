using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction
{
    public class ELuActivator: ELuDerivative, IActivationFunction
    {
        /// <summary>
        /// Create an ELu activator with Alpha 
        /// </summary>
        /// <param name="alpha">A constant to multiply the exp by</param>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public ELuActivator(double alpha): base(alpha) {}
    }
}
