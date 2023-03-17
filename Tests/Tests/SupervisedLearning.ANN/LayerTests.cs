using ANN.Structure.Layer;
using Common.Maths.ActivationFunction.Interface;
using Common.Maths.ActivationFunction;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using SupervisedLearning.ANN.Neuron;

namespace Tests.SupervisedLearning.ANN
{
    [TestFixture]
    public class LayerTests
    {
        private static readonly MatrixBuilder<double> M = Matrix<double>.Build;
        private static readonly IActivationFunction _activator = new LinearActivator();

        [TestCaseSource(nameof(LayerDataSources))]
        public void Create_Should_CreateCorrectly(double[,] weightInputs, int numberOfNeurons, int numberOfWeights)
        {
            var weightsMatrix = M.DenseOfArray(weightInputs);
            var layer = Layer.Create(weightsMatrix, _activator);
            layer.Weights.Should().BeEquivalentTo(M.DenseOfArray(weightInputs));
            layer.Activator.Should().BeEquivalentTo(_activator);
        }

        [TestCaseSource(nameof(LayerDataSources))]
        public void Build_Should_CorrectlyBuildNeurons(double[,] weightInputs, int numberOfNeurons, int numberOfWeights)
        {
            var weightsMatrix = M.DenseOfArray(weightInputs);
            var layer = Layer.Create(weightsMatrix, _activator).Build();

            layer.Neurons.Should().HaveCount(numberOfNeurons);
            var weights = weightInputs.Cast<double>().ToList();
            for (var i = 0; i < numberOfNeurons; i++)
            {
                var neuronWeights = weights.GetRange(i * numberOfWeights, numberOfWeights);
                layer.Neurons[i].Weights.Should().BeEquivalentTo(neuronWeights.GetRange(1, neuronWeights.Count - 1));
                layer.Neurons[i].Bias.Should().Be(neuronWeights[0]);
                layer.Neurons[i].Activator.Should().Be(_activator.Activate);
                layer.Neurons[i].Parents.Should().BeNull();
                layer.Neurons[i].Inputs.Should().BeNull();
            }
        }

        [Test]
        public void Create_Should_ThrowException_When_WeightsAreEmpty()
        {
            Action act = () => Layer.Create(M.DenseOfArray(new double[,] { }), _activator);

            act.Should().Throw<ArgumentException>().WithMessage("Layer must be created with weights");
        }

        public static object[] LayerDataSources =
        {
            new object[] { new double[,] { { 1, 2, 3 } }, 1, 3},
            new object[] { new double[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 }
            }, 3, 3},
            new object[] { new double[,]
            {
                { 1, 2, 3, 0 },
                { 4, 5, 6, 0 },
                { 7, 8, 9, 0 }
            }, 3, 4}
        };
    }
}
