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
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        ann.AddLayer(
                Layer.Create(
                    Matrix<double>.Build.DenseOfArray(
                        new[,]
                        {
                            { 0.13, 0.01, 0.02, 0.03 },
                            { 0.14, 0.04, 0.05, 0.06 },
                            { 0.15, 0.07, 0.08, 0.09 },
                            { 0.16, 0.1, 0.11, 0.12 },
                        }
                    ),
                    IActivationFunction.ActivationFunction.Tanh
                )
            )
            .AddLayer(
                Layer.Create(
                    Matrix<double>.Build.DenseOfArray(
                        new[,]
                        {
                            { 0.25, 0.17, 0.18, 0.19, 0.2 },
                            { 0.26, 0.21, 0.22, 0.23, 0.24 },
                        }
                    ),
                    IActivationFunction.ActivationFunction.Sigmoid
                )
            )
            .SetTrainer(
                new BackPropagationWithGradientDescent(
                    new FlatLearningRate(0.99),
                    new MeanSquaredError(),
                    new SupervisedLearningData(
                        Matrix<double>.Build.DenseOfArray(
                            new double[,]
                            {
                                { 1 },
                                { 2 },
                                { 3 },
                            }
                        ),
                        Matrix<double>.Build.DenseOfArray(
                            new[,]
                            {
                                { 0.25 },
                                { 0.75 },
                            }
                        ),
                        1,
                        0.01
                    ),
                    ann
                )
            )
            .Build();

        ann.Train();
    }
}
