using Common.Maths.ActivationFunction.Derivative;
using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics;

namespace Common.Maths.ActivationFunction
{
    public class SwishActivator: SwishDerivative, IActivationFunction
    {
        /// <summary>
        /// y = x * sigmoid(x)
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Activate(double input)
        {
            return input * SpecialFunctions.Logistic(input);
        }
    }
}
