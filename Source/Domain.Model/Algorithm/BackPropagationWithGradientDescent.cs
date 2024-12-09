using Learning.Supervised.Training.Algorithm.Interface;
using Learning.Supervised.Training.Data;
using Learning.Supervised.Training.LearningRate.Interface;
using Learning.Supervised.Training.LossFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Ann.Algorithm;

public class BackPropagationWithGradientDescent : ITrainer
{
    private readonly Ann _ann;
    private readonly int _batchSize;

    public BackPropagationWithGradientDescent(
        ILearningRate learningRate,
        ILossFunction lossFunction,
        SupervisedLearningData data,
        Ann ann,
        int batchSize
    )
    {
        LearningRate = learningRate;
        LossFunction = lossFunction;
        Data = data;
        _ann = ann;
        _batchSize = batchSize;
    }

    private SupervisedLearningData Data { get; init; }
    private ILearningRate LearningRate { get; init; }
    private ILossFunction LossFunction { get; init; }

    public void Train()
    {
        try
        {
            if (!_ann.HasBeenBuilt)
                throw new InvalidOperationException("Cannot train Ann before building.");

            var batchCount = 0;
            var batchLoss = 0.0;
            for (var epoch = 0; epoch < Data.MaxEpochs; epoch++)
            {
                var loss = TrainOnce(epoch);

                batchLoss += loss;
                if (batchCount == _batchSize)
                {
                    batchCount = 0;
                    if (batchLoss / _batchSize < Data.MinError)
                        return;
                    batchLoss = 0;
                }
            }
        }
        catch (Exception e)
        {
            throw new Exception("Error running the Ann during gradient descent: ", e);
        }
    }

    private double TrainOnce(int epoch)
    {
        const double momentum = 0.01;

        var (inputs, expectedOutputs) = Data.GetInputsOutputs(epoch);

        _ann.Run(inputs);

        // Perform gradient descent (via backprop)
        var currentLayer = _ann.Layers.Last();

        // Iterate backwards through all the layers
        // Calculate all gradients
        while (currentLayer?.Inputs is not null)
        {
            // Calculate the sigmas (error term)
            Vector<double> sigmas;
            if (currentLayer.OutputLayer is null || currentLayer.Outputs is null)
            {
                // Output layer has a different sigma calculation
                sigmas = expectedOutputs.Subtract(_ann.Outputs);
            }
            else
            {
                var downstreamLayerCount = currentLayer.OutputLayer.Outputs!.Count;
                var thisLayerCount = currentLayer.Outputs!.Count;

                // Builds a vector [0.0 0.0 ... 0.0] to init sigmas to 0.0 for the length of this layer
                sigmas = Vector<double>.Build.Dense(thisLayerCount, 0.0);

                // Iterate through each downstream gradient
                for (var row = 0; row < downstreamLayerCount; row++)
                {
                    // Create a vector [grad_O1 grad_02 ... grad_0n] with length of the current layer
                    var gradientVector = Vector<double>.Build.Dense(
                        currentLayer.Outputs.Count,
                        currentLayer.OutputLayer.Gradients![row]
                    );

                    // sigma_n_i = sigma_n_i-1 + (grad_Oi+1 * weight_n_i)
                    sigmas = sigmas.Add(
                        gradientVector.PointwiseMultiply(
                            currentLayer
                                .OutputWeights!.Row(row)
                                .SubVector(0, currentLayer.Outputs.Count)
                        )
                    );
                }
            }

            // Calculate the gradients for the neurons in this layer
            currentLayer.Gradients = currentLayer.Derivatives!.PointwiseMultiply(sigmas);

            currentLayer = currentLayer.InputLayer;
        }

        currentLayer = _ann.Layers.Last();

        // Iterate backwards through all the layers
        // Calculate deltas and update weights
        while (currentLayer?.Inputs is not null)
        {
            // Get the old deltas (or init to 0s)
            var currentDeltas =
                currentLayer.Deltas
                ?? Matrix<double>.Build.Dense(
                    currentLayer.InputWeights!.RowCount,
                    currentLayer.InputWeights.ColumnCount,
                    0.0
                );

            // Calculate delta matrix
            var rowDeltas = new List<Vector<double>>();
            for (var row = 0; row < currentLayer.InputWeights!.RowCount; row++)
            {
                rowDeltas.Add(
                    currentLayer.Inputs.Multiply(LearningRate.Apply(currentLayer.Gradients![row]))
                );
            }

            currentLayer.Deltas = Matrix<double>.Build.DenseOfRowVectors(rowDeltas);

            // Update weights
            currentLayer.InputWeights = currentLayer
                .Deltas.Add(currentLayer.InputWeights)
                .Add(currentDeltas.Multiply(momentum));

            currentLayer = currentLayer.InputLayer;
        }

        return LossFunction.CalculateLoss(expectedOutputs, _ann.Outputs);
    }
}
