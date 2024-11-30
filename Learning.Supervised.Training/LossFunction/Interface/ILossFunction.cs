using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Training.LossFunction.Interface;

public interface ILossFunction
{
    /// <summary>
    /// Returns the vector loss for a vector output, expecting a vector output
    /// </summary>
    /// <param name="expected"></param>
    /// <param name="actual"></param>
    /// <returns></returns>
    public Vector<double> CalculateLoss(Vector<double> expected, Vector<double> actual);
}
