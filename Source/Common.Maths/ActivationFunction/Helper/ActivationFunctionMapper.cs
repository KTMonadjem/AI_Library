using System.ComponentModel;
using Common.Maths.ActivationFunction.Interface;

namespace Common.Maths.ActivationFunction.Helper;

public static class ActivationFunctionMapper
{
    /// <summary>
    ///     Maps an activation function enum to a new activator
    /// </summary>
    /// <param name="activationFunction"></param>
    /// <param name="alpha">Alpha parameter required</param>
    /// <returns></returns>
    public static IActivationFunction MapActivationFunction(IActivationFunction.ActivationFunction activationFunction,
        double? alpha = null)
    {
        return activationFunction switch
        {
            IActivationFunction.ActivationFunction.Binary => new BinaryActivator(),
            IActivationFunction.ActivationFunction.Linear => new LinearActivator(),
            IActivationFunction.ActivationFunction.ReLu => new ReLuActivator(),
            IActivationFunction.ActivationFunction.LeakyReLu => new LeakyReLuActivator(alpha ?? 0.01),
            IActivationFunction.ActivationFunction.ELu => new ELuActivator(alpha ?? 1.0),
            IActivationFunction.ActivationFunction.Sigmoid => new SigmoidActivator(),
            IActivationFunction.ActivationFunction.Tanh => new TanhActivator(),
            IActivationFunction.ActivationFunction.Swish => new SwishActivator(),
            _ => throw new InvalidEnumArgumentException()
        };
    }
}