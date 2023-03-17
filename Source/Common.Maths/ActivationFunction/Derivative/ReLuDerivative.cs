using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction.Derivative
{
    public class ReLuDerivative: IActivationDerivative
    {
        /// <summary>
        /// y' = 1 if x >= 0
        /// y' = 0 if x < 0
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derive(double x)
        {
            return x >= 0 ? 1 : 0;
        }
    }
}
