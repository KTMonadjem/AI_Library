using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction
{
    public class BinaryActivator: BinaryDerivative, IActivationFunction
    {
        public double Delta { get; set; }

        /// <summary>
        /// y = 1 if x > 0
        /// y = 0 if x <= 0
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Activate(double input)
        {
            Delta = Derive(input);
            return input > 0 ? 1 : 0;
        }
    }
}
