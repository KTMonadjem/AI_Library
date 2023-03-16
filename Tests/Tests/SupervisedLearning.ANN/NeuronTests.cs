using Common.Maths.ActivationFunction;
using SupervisedLearning.ANN.Neuron;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using System.Runtime.CompilerServices;
using Common.Maths.ActivationFunction.Interface;

namespace Tests.Domain.Model
{
    [TestFixture]
    public class NeuronTests
    {
        private static readonly Vector<double> _inputs = 
            Vector<double>.Build.Dense(new double[] { 0, 1, 2, 3, 4, 5 });
        private static readonly Vector<double> _weights = _inputs.Clone();
        private const double Bias = 10;
        private static readonly IActivationFunction _activator = new LinearActivator();

        [Test]
        public void CreateNeuron_Should_CreateWithCorrectValues()
        {
            var neuron = Neuron.Create(_inputs, _weights, Bias, _activator.Activate);
            neuron.Inputs.Should().BeEquivalentTo(_inputs);
            neuron.Weights.Should().BeEquivalentTo(_weights);
            neuron.Bias.Should().Be(Bias);
            neuron.Activator.Should().BeEquivalentTo(_activator.Activate);
        }

        [Test]
        public void Neuron_Should_ActivateCorrectly()
        {
            var neuron = Neuron.Create(_inputs, _weights, Bias, _activator.Activate);
            neuron.Activate();
            var output = neuron.Output;

            // 1 + 4 + 9 + 16 + 25 + 10 = 65
            output.Should().Be(65);
        }
    }
}
