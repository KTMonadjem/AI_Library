using Common.Maths.ActivationFunction;
using FluentAssertions;
using Learning.Supervised.ArtificialNeuralNetwork;
using Learning.Supervised.ArtificialNeuralNetwork.Algorithm;
using Learning.Supervised.ArtificialNeuralNetwork.Structure;
using Learning.Supervised.Training.Data;
using Learning.Supervised.Training.LearningRate;
using Learning.Supervised.Training.LossFunction;
using MathNet.Numerics.LinearAlgebra;

namespace Tests.Supervised.Learning.ArtificialNeuralNetwork;

public class BackPropagationWithGradientDescentTests
{
    [Test]
    public void BackPropagationWithGradientDescent_Should_ConsistentlyTrainXOR()
    {
        var trainingData = new SupervisedLearningData(
            Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { 0, 1, 1, 0 },
                    { 1, 0, 1, 0 },
                }
            ),
            Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { 1, 1, 0, 0 },
                }
            ),
            1000,
            0.01
        );
        var ann = Ann.Create()
            .AddLayer(
                Layer.Create(
                    Matrix<double>.Build.DenseOfArray(
                        new[,]
                        {
                            { 0.45057174000671785, 0.09108144327657153, 0.512013203591166 },
                            { 0.35758729385152177, 0.6962409058332407, 0.1358240703712147 },
                        }
                    ),
                    new ReLuActivator()
                )
            )
            .AddLayer(
                Layer.Create(
                    Matrix<double>.Build.DenseOfArray(
                        new[,]
                        {
                            { 0.24165547530971054, 0.6096661159973799, 0.9948427235898957 },
                        }
                    ),
                    new ReLuActivator()
                )
            )
            .Build();

        var trainer = new BackPropagationWithGradientDescent(
            new FlatLearningRate(0.2),
            new MeanSquaredError(),
            trainingData,
            ann,
            4
        );

        var trainingOutput = trainer.Train();

        trainingOutput.Loss.Should().BeApproximately(0.008636684851844992, 0.000001);
        trainingOutput.Epochs.Should().Be(435);

        var expectedFirstLayerWeights = new[,]
        {
            { -0.7162031830730357, -0.7318886414318757, 0.7297514147992815 },
            { 0.8082973495677143, 0.8003827801904254, -0.814651231775803 },
        };

        var firstLayer = ann.Layers.First();
        for (var weightRow = 0; weightRow < firstLayer.InputWeights!.RowCount; weightRow++)
        {
            var nthRow = firstLayer.InputWeights!.Row(weightRow);
            for (var weightCol = 0; weightCol < firstLayer.InputWeights!.ColumnCount; weightCol++)
            {
                nthRow[weightCol]
                    .Should()
                    .BeApproximately(expectedFirstLayerWeights[weightRow, weightCol], 0.000001);
            }
        }

        var expectedSecondLayerWeights = new[,]
        {
            { -1.174265967213254, -1.1614324909510871, 0.9007204642910601 },
        };

        var secondLayer = firstLayer.OutputLayer;
        for (var weightRow = 0; weightRow < secondLayer!.InputWeights!.RowCount; weightRow++)
        {
            var nthRow = secondLayer.InputWeights!.Row(weightRow);
            for (var weightCol = 0; weightCol < secondLayer.InputWeights!.ColumnCount; weightCol++)
            {
                nthRow[weightCol]
                    .Should()
                    .BeApproximately(expectedSecondLayerWeights[weightRow, weightCol], 0.000001);
            }
        }

        var expectedOutputs = new[,]
        {
            { 0.9007204642910601 },
            { 0.8848112368590108 },
            { 0.0 },
            { 0.04379821336654122 },
        };
        for (var i = 0; i < trainingData.NumberOfInputs; i++)
        {
            var (inputs, _) = trainingData.GetInputsOutputs(i);

            ann.Run(inputs);

            ann.Outputs[0].Should().BeApproximately(expectedOutputs[i, 0], 0.000001);
        }
    }
}
