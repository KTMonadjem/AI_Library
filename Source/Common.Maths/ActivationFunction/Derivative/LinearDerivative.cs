using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction.Derivative
{
    public class LinearDerivative: IActivationDerivative
    {
        public double Evaluate(double x) 
        {
            return x;
        }
    }
}
