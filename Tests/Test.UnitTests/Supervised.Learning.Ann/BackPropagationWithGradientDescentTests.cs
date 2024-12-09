using Common.Maths.ActivationFunction;
using Common.Maths.ActivationFunction.Interface;
using Learning.Supervised.Ann.Algorithm;
using Learning.Supervised.Ann.Structure;
using Learning.Supervised.Training.Data;
using Learning.Supervised.Training.LearningRate;
using Learning.Supervised.Training.LossFunction;
using MathNet.Numerics.LinearAlgebra;

namespace Tests.Supervised.Learning.Ann;

public class BackPropagationWithGradientDescentTests
{
    [Test]
    public void BackPropagationWithGradientDescent_()
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
            100,
            0.01
        );
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        ann.AddLayer(
                Layer.Create(
                    Matrix<double>.Build.DenseOfArray(
                        new[,]
                        {
                            { 0.1, 0.2, 0.3 },
                            { 0.4, 0.5, 0.6 },
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
                            { 0.7, 0.8, 0.9 },
                        }
                    ),
                    new SigmoidActivator()
                )
            )
            .SetTrainer(
                new BackPropagationWithGradientDescent(
                    new FlatLearningRate(0.9),
                    new MeanSquaredError(),
                    trainingData,
                    ann
                )
            )
            .Build();

        ann.Train();

        for (var i = 0; i < trainingData.NumberOfInputs; i++)
        {
            var (inputs, outputs) = trainingData.GetInputsOutputs(i);

            ann.Run(inputs);

            Console.WriteLine(
                $"Inputs: [{string.Join(",", inputs)}] -> Outputs: [{string.Join(",", ann.Outputs)}]"
            );
        }
    }
}
