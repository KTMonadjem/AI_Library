using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction
{
    public class TanhActivator : TanhDerivative, IActivationFunction
    {
        /// <summary>
        /// y = tanh(x)
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Activate(double input)
        {
            return Trig.Tanh(input);
        }
    }
}
