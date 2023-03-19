using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction.Derivative
{
    public class SigmoidDerivative: IActivationDerivative
    {
        protected double SigmoidX;

        /// <summary>
        /// y' = sigmoid(x) * (1 - sigmoid(x))
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derive(double x)
        {
            var log = SigmoidX;
            return log * (1 - log);
        }
    }
}
