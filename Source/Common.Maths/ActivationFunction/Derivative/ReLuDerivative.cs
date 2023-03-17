using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction.Derivative
{
    public class ReLuDerivative: IActivationDerivative
    {
        public double Evaluate(double x)
        {
            return x >= 0 ? 1 : 0;
        }
    }
}
