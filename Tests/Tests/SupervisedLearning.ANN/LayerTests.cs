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
        private static readonly VectorBuilder<double> V = Vector<double>.Build;
        private static readonly IActivationFunction _activator = new LinearActivator();

        private const int _numberOfNeurons = 10;
        private const int _numberOfWeights = 20;
        private const double _minWeight = -0.5;
        private const double _maxWeight = 0.75;

        [TestCaseSource(nameof(LayerDataSources))]
        public void Create_Should_CreateCorrectly(double[,] weightInputs, int numberOfNeurons, int numberOfWeights)
        {
            var weightsMatrix = M.DenseOfArray(weightInputs);
            var layer = Layer.Create(weightsMatrix, _activator);
            layer.Weights.Should().BeEquivalentTo(M.DenseOfArray(weightInputs));
            layer.Activator.Should().BeEquivalentTo(_activator);
        }

        [TestCase(_numberOfNeurons, _numberOfWeights, _minWeight, _maxWeight)]
        [TestCase(100, 200, 0, 1)]
        [TestCase(1, 1, 0, 10)]
        public void CreateWithRandomWeights_Should_CreateWithRandomWeightsCorrectly(int numberOfNeurons, int numberOfWeights, double minWeight, double maxWeight)
        {
            var layer = Layer.CreateWithRandomWeights(numberOfNeurons, numberOfWeights, minWeight, maxWeight, _activator);

            layer.BuildWeights();
            layer.Neurons.Should().HaveCount(numberOfNeurons);
            foreach (var neuron in layer.Neurons)
            {
                neuron.Weights.Should().HaveCount(numberOfWeights);
                neuron.Bias.Should().BeInRange(minWeight, maxWeight);
                foreach (var weight in neuron.Weights)
                {
                    weight.Should().BeInRange(minWeight, maxWeight);
                }
            }
            layer.Activator.Should().BeEquivalentTo(_activator);
        }

        [TestCase(0)]
        [TestCase(-10)]
        public void CreateWithRandomWeights_Should_ThrowException_When_TooFewNeurons(int numberOfNeurons)
        {
            Action act = () => Layer.CreateWithRandomWeights(numberOfNeurons, _numberOfWeights, _minWeight, _maxWeight, _activator);

            act.Should().Throw<ArgumentException>().WithMessage("Layer must be created with neurons");
        }

        [TestCase(0)]
        [TestCase(-10)]
        public void CreateWithRandomWeights_Should_ThrowException_When_TooFewWeights(int numberOfWeights)
        {
            Action act = () => Layer.CreateWithRandomWeights(_numberOfNeurons, numberOfWeights, _minWeight, _maxWeight, _activator);

            act.Should().Throw<ArgumentException>().WithMessage("Layer must be created with weights");
        }

        [TestCase(5, 0)]
        [TestCase(0.1, -0.1)]
        public void CreateWithRandomWeights_Should_ThrowException_When_MinWeightLessThanMaxWeight(double minWeight, double maxWeight)
        {
            Action act = () => Layer.CreateWithRandomWeights(_numberOfNeurons, _numberOfWeights, minWeight, maxWeight, _activator);

            act.Should().Throw<ArgumentException>().WithMessage("Min weight must be less than max weight");
        }

        [TestCaseSource(nameof(LayerDataSources))]
        public void BuildWeights_Should_CorrectlyBuildNeuronWeights(double[,] weightInputs, int numberOfNeurons, int numberOfWeights)
        {
            var weightsMatrix = M.DenseOfArray(weightInputs);
            var layer = Layer.Create(weightsMatrix, _activator).BuildWeights();

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

        [Test]
        public void AddInputs_Should_ThrowException_When_NoInputsProvided()
        {
            var layer = Layer.CreateWithRandomWeights(_numberOfNeurons, _numberOfWeights, _minWeight, _maxWeight, _activator).BuildWeights();
            Action act = () => layer.AddInputs(V.DenseOfArray(new double[] { }));

            act.Should().Throw<ArgumentException>().WithMessage("Must have at least one input");
        }

        [Test]
        public void AddInputs_Should_Succeed_When_InputsAreValid()
        {
            var inputs = new double[_numberOfNeurons];
            for (var i = 0; i < _numberOfNeurons; i++)
            {
                inputs[i] = i;
            }
            var layer = Layer.CreateWithRandomWeights(_numberOfNeurons, _numberOfNeurons, _minWeight, _maxWeight, _activator)
                .BuildWeights()
                .AddInputs(V.DenseOfArray(inputs));

            foreach (var neuron in layer.Neurons)
            {
                neuron.Inputs!.Should().BeEquivalentTo(inputs);
            }
        }

        [Test]
        public void AddParents_Should_ThrowException_When_NoParentsProvided()
        {
            var layer = Layer.CreateWithRandomWeights(_numberOfNeurons, _numberOfWeights, _minWeight, _maxWeight, _activator).BuildWeights();
            Action act = () => layer.AddParents(Array.Empty<Neuron>().ToList());

            act.Should().Throw<ArgumentException>().WithMessage("Parents must be provided to add parents to this layer");
        }

        [Test]
        public void AddParents_Should_Succeed_When_ValidParents()
        {
            var weights = Vector<double>.Build.Dense(new double[] { });
            var parents = new List<Neuron>
            {
                Neuron.Create(weights, 0, _activator.Activate),
                Neuron.Create(weights.Multiply(2), 0, _activator.Activate)
            };
            var layer = Layer.CreateWithRandomWeights(parents.Count, parents.Count, 0, 1, _activator)
                .BuildWeights()
                .AddParents(parents);

            foreach (var neuron in layer.Neurons)
            {
                neuron.Parents.Should().BeEquivalentTo(parents);
            }
        }

        [Test]
        public void AddParentLayer_Should_Succeed_When_ValidParents()
        {
            var parentCount = 2;
            var parents = Layer.CreateWithRandomWeights(parentCount, parentCount, 0, 1, _activator)
                .BuildWeights();
            var layer = Layer.CreateWithRandomWeights(parentCount, parentCount, 0, 1, _activator)
                .BuildWeights()
                .AddParentLayer(parents);

            foreach (var neuron in layer.Neurons)
            {
                neuron.Parents.Should().BeEquivalentTo(parents.Neurons);
            }
        }

        [Test]
        public void Clone_Should_CloneTheLayer()
        {
            var layer = Layer.CreateWithRandomWeights(_numberOfNeurons, _numberOfWeights, _minWeight, _maxWeight, _activator).BuildWeights();
            var clone = layer.Clone();

            clone.Weights.Should().BeEquivalentTo(layer.Weights);
            clone.Activator.Should().BeEquivalentTo(layer.Activator);
            clone.Neurons.Should().BeEmpty();
            clone.IsBuilt.Should().BeFalse();
            clone.HasInputs.Should().BeFalse();
        }
    }
}
