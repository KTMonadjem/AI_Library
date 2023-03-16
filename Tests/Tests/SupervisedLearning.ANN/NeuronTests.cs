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
            Vector<double>.Build.Dense(new double[] { 1, 2, 3, 4, 5 });

        private static readonly Vector<double> _parent1Inputs = Vector<double>.Build.Dense(new double[] { 1, 2 });
        private static readonly Vector<double> _parent1Weights = _parent1Inputs.Clone();
        private static readonly Vector<double> _parent2Inputs = Vector<double>.Build.Dense(new double[] { 3, 4 });
        private static readonly Vector<double> _parent2Weights = _parent2Inputs.Clone();
        private static List<Neuron> _parents;

        private static readonly Vector<double> _weights = _inputs.Clone();
        private const double Bias = 10;
        private static readonly IActivationFunction _activator = new LinearActivator();

        [SetUp]
        public void Setup()
        {
            var parent1 = Neuron.CreateWithInputs(_parent1Inputs, _parent1Weights, Bias, _activator.Activate);
            var parent2 = Neuron.CreateWithInputs(_parent2Inputs, _parent2Weights, Bias, _activator.Activate);
            _parents = new List<Neuron>
            {
                parent1, 
                parent2
            };
        }

        [Test]
        public void CreateWithInputs_Should_CreateWithCorrectValues()
        {
            var neuron = Neuron.CreateWithInputs(_inputs, _weights, Bias, _activator.Activate);
            neuron.Inputs.Should().BeEquivalentTo(_inputs);
            neuron.Parents.Should().BeNull();
            neuron.Weights.Should().BeEquivalentTo(_weights);
            neuron.Bias.Should().Be(Bias);
            neuron.Activator.Should().BeEquivalentTo(_activator.Activate);
        }

        [Test]
        public void CreateWithParents_Should_CreateWithCorrectValues()
        {
            var weights = _weights.SubVector(0, _parents.Count);
            var neuron = Neuron.CreateWithParents(_parents, weights, Bias, _activator.Activate);

            neuron.Inputs.Should().BeNull();
            neuron.Parents.Should().BeEquivalentTo(_parents);
            neuron.Weights.Should().BeEquivalentTo(weights);
            neuron.Bias.Should().Be(Bias);
            neuron.Activator.Should().BeEquivalentTo(_activator.Activate);
        }

        [Test]
        public void NeuronWithInputs_Should_ActivateCorrectly()
        {
            var neuron = Neuron.CreateWithInputs(_inputs, _weights, Bias, _activator.Activate);
            var output = neuron.Output;

            var expected = Bias;
            for (var i = 0; i < _inputs.Count; i++)
            {
                expected += _inputs[i] * _weights[i];
            }
            expected = _activator.Activate(expected);

            output.Should().Be(expected);
        }

        [Test]
        public void NeuronWithParents_Should_ActivateCorrectly()
        {
            var weights = _weights.SubVector(0, _parents.Count);
            var neuron = Neuron.CreateWithParents(_parents, weights, Bias, _activator.Activate);
            var output = neuron.Output;

            var expected = Bias;
            for (var i = 0; i < _parents.Count; i++)
            {
                expected += _parents[i].Output * weights[i];
            }
            expected = _activator.Activate(expected);

            output.Should().Be(expected);
        }
    }
}
