using Data.DataSet.Interface;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Learning.Supervised.Training.Data
{
    public class SupervisedLearningData: IDataSet
    {
        private readonly Matrix<double> _inputs;
        private readonly Matrix<double> _outputs;

        public SupervisedLearningData(Matrix<double> inputs, Matrix<double> outputs)
        {
            _inputs = inputs;
            _outputs = outputs;
        }

        public Matrix<double> GetInputsOutputs(int column)
        {
            if (_inputs.ColumnCount <= column)
            {
                throw new ArgumentOutOfRangeException(nameof(column));
            }
            if (_outputs.ColumnCount <= column)
            {
                throw new ArgumentOutOfRangeException(nameof(column));
            }
            return Matrix<double>.Build.DenseOfColumns(new List<Vector<double>> { _inputs.Column(column), _outputs.Column(column) });
        }
    }
}
