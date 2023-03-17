using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction.Derivative
{
    public class LeakyReLuDerivative: IActivationDerivative
    {
        private readonly double _leak;

        public LeakyReLuDerivative(double leak)
        {
            if (leak < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(leak));
            }

            _leak = leak;
        }

        public double Evaluate(double x)
        {
            return x >= 0 ? 1 : _leak;
        }
    }
}
