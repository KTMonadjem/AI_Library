using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data.DataSet.Interface
{
    public interface IDataSet
    {
        public Matrix<double> GetInputsOutputs(int column);
    }
}
