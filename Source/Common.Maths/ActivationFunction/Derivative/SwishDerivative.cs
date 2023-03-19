using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction.Derivative
{
    public class SwishDerivative: IActivationDerivative
    {
        protected double SigmoidX;
        protected double SwishX;

        /// <summary>
        /// y' = x * sigmoid(x) + sigmoid(x)(1 - x * sigmoid(x))
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derive(double x)
        {
            return SwishX + SigmoidX * (1 - SwishX);
        }
    }
}
