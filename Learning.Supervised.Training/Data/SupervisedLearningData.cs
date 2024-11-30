using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Training.Data;

public class SupervisedLearningData
{
    private readonly Matrix<double> _inputs;
    private readonly Matrix<double> _outputs;
    private readonly bool _canLoopData;

    public SupervisedLearningData(
        Matrix<double> inputs,
        Matrix<double> outputs,
        int? maxEpochs = null,
        double? minError = null
    )
    {
        if (inputs.ColumnCount != outputs.ColumnCount)
            throw new ArgumentException("Inputs and outputs must have the same number of columns");

        _inputs = inputs;
        _outputs = outputs;
        MaxEpochs = maxEpochs ?? inputs.ColumnCount;
        MinError = minError ?? 0;

        _canLoopData = MaxEpochs > inputs.ColumnCount;
    }

    public int MaxEpochs { get; private init; }
    public double MinError { get; private init; }

    public (Vector<double> Inputs, Vector<double> Outputs) GetInputsOutputs(int column)
    {
        if (_canLoopData)
            column %= _inputs.ColumnCount;

        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, _inputs.ColumnCount);

        return (_inputs.Column(column), _outputs.Column(column));
    }
}
