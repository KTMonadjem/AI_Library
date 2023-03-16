using ANN.Structure.Layer;
using Common.Maths.ActivationFunction.Interface;
using Common.Maths.ActivationFunction;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;

namespace Tests.SupervisedLearning.ANN
{
    [TestFixture]
    public class LayerTests
    {
        private static readonly MatrixBuilder<double> M = Matrix<double>.Build;

        private static readonly double[,] _weightInputs = new double[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 }
            };
        private static Matrix<double> _weights;
        private static readonly IActivationFunction _activator = new LinearActivator();

        [SetUp]
        public void SetUp()
        {
            _weights = M.DenseOfArray(_weightInputs);
        }

        [Test]
        public void Create_Should_CreateCorrectly()
        {
            var layer = Layer.Create(_weights, _activator);
            layer.Weights.Should().BeEquivalentTo(M.DenseOfArray(_weightInputs));
            layer.Activator.Should().BeEquivalentTo(_activator);
        }

        [Test]
        public void Build_Should_CorrectlyBuildNeurons()
        {
            var layer = Layer.Create(_weights, _activator).Build();

            layer.Neurons.Should().HaveCount(3);
            var weights = _weightInputs.Cast<double>().ToList();
            var weightsPerNeuron = weights.Count / 3;
            for (var i = 0; i < 3; i++)
            {
                var neuronWeights = weights.GetRange(i * 3, weightsPerNeuron);
                layer.Neurons[i].Weights.Should().BeEquivalentTo(neuronWeights.GetRange(1, neuronWeights.Count - 1));
                layer.Neurons[i].Bias.Should().Be(neuronWeights[0]);
                layer.Neurons[i].Activator.Should().Be(_activator.Activate);
                layer.Neurons[i].Parents.Should().BeNull();
                layer.Neurons[i].Inputs.Should().BeNull();
            }
        }
    }
}
