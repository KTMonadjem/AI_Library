using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction.Derivative
{
    public class TanhDerivative: IActivationDerivative
    {
        /// <summary>
        /// y' = 1 - tanh^2(x)
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derive(double x)
        {
            var tanh = Trig.Tanh(x);
            return 1 - Math.Pow(tanh, 2);
        }
    }
}
