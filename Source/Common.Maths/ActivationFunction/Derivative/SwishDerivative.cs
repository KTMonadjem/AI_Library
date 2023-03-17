using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction.Derivative
{
    public class SwishDerivative: IActivationDerivative
    {
        /// <summary>
        /// y' = x * sigmoid(x) + sigmoid(x)(1 - x * sigmoid(x))
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derive(double x)
        {
            var sigmoid = SpecialFunctions.Logistic(x);
            return x * sigmoid + sigmoid * (1 - x * sigmoid);
        }
    }
}
