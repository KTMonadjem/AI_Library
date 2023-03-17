using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction.Derivative
{
    public class SigmoidDerivative: IActivationDerivative
    {
        /// <summary>
        /// y' = sigmoid(x) * (1 - sigmoid(x))
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derive(double x)
        {
            var log = SpecialFunctions.Logistic(x);
            return log * (1 - log);
        }
    }
}
