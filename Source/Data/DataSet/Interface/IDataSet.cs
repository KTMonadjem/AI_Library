using MathNet.Numerics.LinearAlgebra;

namespace Data.DataSet.Interface;

public interface IDataSet
{
    public Matrix<double> GetInputsOutputs(int column);
}