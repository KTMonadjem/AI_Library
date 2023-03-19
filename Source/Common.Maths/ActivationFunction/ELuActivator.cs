using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction
{
    public class ELuActivator: ELuDerivative, IActivationFunction
    {
        public double Delta { get; set; }

        /// <summary>
        /// Create an ELu activator with Alpha 
        /// </summary>
        /// <param name="alpha">A constant to multiply the exp by</param>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public ELuActivator(double alpha): base(alpha) {}

        /// <summary>
        /// y = x if x > 0
        /// y = Alpha * (e^x - 1)
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Activate(double input)
        {
            Beta = input > 0 ? input : _alpha * (Math.Pow(Math.E, input) - 1);
            Delta = Derive(input);
            return Beta;
        }
    }
}
