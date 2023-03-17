using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction
{
    public class SigmoidActivator: SigmoidDerivative, IActivationFunction
    {
        /// <summary>
        /// y = sigmoid(x)
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Activate(double input)
        {
            return SpecialFunctions.Logistic(input);
        }
    }
}
