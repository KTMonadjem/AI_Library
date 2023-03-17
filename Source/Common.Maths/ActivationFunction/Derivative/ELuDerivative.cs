using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction.Derivative
{
    public class ELuDerivative: IActivationDerivative
    {
        private readonly double _alpha;

        /// <summary>
        /// Create a new ELu Derivative with alpha (from the activator)
        /// </summary>
        /// <param name="alpha">A constant to multiply the exp by</param>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public ELuDerivative(double alpha)
        {
            if (alpha < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(alpha));
            }

            _alpha = alpha;
        }

        /// <summary>
        /// y' = 1 if x > 0
        /// y' = Alpha * (e^x - 1) + Alpha
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derive(double x)
        {
            return x >= 0 ? 1 : Activate(x) + _alpha;
        }

        /// <summary>
        /// y = x if x > 0
        /// y = Alpha * (e^x - 1)
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Activate(double input)
        {
            return input > 0 ? input : _alpha * (Math.Pow(Math.E, input) - 1);
        }
    }
}
