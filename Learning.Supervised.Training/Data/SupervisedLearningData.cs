using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Training.Data;

public class SupervisedLearningData
{
    private readonly Matrix<double> _inputs;
    private readonly Matrix<double> _outputs;

    public SupervisedLearningData(
        Matrix<double> inputs,
        Matrix<double> outputs,
        int maxEpochs,
        double minError
    )
    {
        _inputs = inputs;
        _outputs = outputs;
        MaxEpochs = maxEpochs;
        MinError = minError;
    }

    public int MaxEpochs { get; private init; }
    public double MinError { get; private init; }

    public Matrix<double> GetInputsOutputs(int column)
    {
        if (_inputs.ColumnCount <= column)
            throw new ArgumentOutOfRangeException(nameof(column));
        if (_outputs.ColumnCount <= column)
            throw new ArgumentOutOfRangeException(nameof(column));
        return Matrix<double>.Build.DenseOfColumns(
            new List<Vector<double>> { _inputs.Column(column), _outputs.Column(column) }
        );
    }
}
