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
                            { 0.7, 0.8 },
                        }
                    ),
                    new SigmoidActivator()
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
                                { 0, 0 },
                                { 0, 1 },
                                { 1, 0 },
                                { 1, 1 },
                            }
                        ),
                        Matrix<double>.Build.DenseOfArray(
                            new double[,]
                            {
                                { 0 },
                                { 1 },
                                { 1 },
                                { 0 },
                            }
                        ),
                        100,
                        0.01
                    ),
                    ann
                )
            )
            .Build();

        ann.Train();
    }
}
