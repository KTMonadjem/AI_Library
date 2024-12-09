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
            1000,
            0.01
        );
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        ann.AddLayer(Layer.CreateWithRandomWeights(3, new SigmoidActivator()))
            .AddLayer(Layer.CreateWithRandomWeights(4, new SigmoidActivator()))
            .AddLayer(Layer.CreateWithRandomWeights(1, new SigmoidActivator()))
            .SetNumberOfInputs(2)
            .SetTrainer(
                new BackPropagationWithGradientDescent(
                    new FlatLearningRate(0.01),
                    new MeanSquaredError(),
                    trainingData,
                    ann,
                    4
                )
            )
            .Build();

        ann.Train();

        for (var i = 0; i < trainingData.NumberOfInputs; i++)
        {
            var (inputs, _) = trainingData.GetInputsOutputs(i);

            ann.Run(inputs);

            Console.WriteLine(
                $"Inputs: [{string.Join(",", inputs)}] -> Outputs: [{string.Join(",", ann.Outputs)}]"
            );
        }
    }
}
