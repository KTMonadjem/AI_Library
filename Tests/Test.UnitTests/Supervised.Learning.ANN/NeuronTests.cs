using Common.Maths.ActivationFunction;
using Common.Maths.ActivationFunction.Interface;
using FluentAssertions;
using Learning.Supervised.ANN.Structure;
using MathNet.Numerics.LinearAlgebra;

namespace Tests.Supervised.Learning.ANN.Structure;

[TestFixture]
public class NeuronTests
{
    [SetUp]
    public void Setup()
    {
        var parent1 = Neuron.CreateWithInputs(_parent1Inputs, _parent1Weights, Bias, _activator);
        var parent2 = Neuron.CreateWithInputs(_parent2Inputs, _parent2Weights, Bias, _activator);
        _parents = new List<Neuron>
        {
            parent1,
            parent2
        };
    }

    private static readonly VectorBuilder<double> V = Vector<double>.Build;

    private static readonly Vector<double> _inputs =
        Vector<double>.Build.Dense(new double[] { 1, 2, 3, 4, 5 });

    private static readonly Vector<double> _parent1Inputs = V.Dense(new double[] { 1, 2 });
    private static readonly Vector<double> _parent1Weights = _parent1Inputs.Clone();
    private static readonly Vector<double> _parent2Inputs = V.Dense(new double[] { 3, 4 });
    private static readonly Vector<double> _parent2Weights = _parent2Inputs.Clone();
    private static List<Neuron> _parents;

    private static readonly Vector<double> _weights = _inputs.Clone();
    private const double Bias = 10;
    private static readonly IActivationFunction _activator = new LinearActivator();

    [Test]
    public void Create_Should_CreateWithCorrectValues()
    {
        var neuron = Neuron.Create(_weights, Bias, _activator);
        neuron.Inputs.Should().BeNull();
        neuron.Parents.Should().BeNull();
        neuron.Weights.Should().BeEquivalentTo(_weights);
        neuron.Bias.Should().Be(Bias);
        neuron.Activator.Should().BeEquivalentTo(_activator);
    }

    [Test]
    public void CreateWithInputs_Should_CreateWithCorrectValues_When_InputsAreValid()
    {
        var neuron = Neuron.CreateWithInputs(_inputs, _weights, Bias, _activator);
        neuron.Inputs.Should().BeEquivalentTo(_inputs);
        neuron.Parents.Should().BeNull();
        neuron.Weights.Should().BeEquivalentTo(_weights);
        neuron.Bias.Should().Be(Bias);
        neuron.Activator.Should().BeEquivalentTo(_activator);
    }

    [Test]
    public void CreateWithInputs_Should_ThrowException_When_InputsAreInvalid()
    {
        Action act = () =>
            Neuron.CreateWithInputs(_inputs, _weights.SubVector(1, _weights.Count - 1), Bias, _activator);

        act.Should().Throw<ArgumentException>().WithMessage("Neuron inputs and weights must be the same length");
    }

    [Test]
    public void CreateWithParents_Should_CreateWithCorrectValues_When_InputsAreValid()
    {
        var weights = _weights.SubVector(0, _parents.Count);
        var neuron = Neuron.CreateWithParents(_parents, weights, Bias, _activator);

        neuron.Inputs.Should().BeNull();
        neuron.Parents.Should().BeEquivalentTo(_parents);
        neuron.Weights.Should().BeEquivalentTo(weights);
        neuron.Bias.Should().Be(Bias);
        neuron.Activator.Should().BeEquivalentTo(_activator);
    }

    [Test]
    public void CreateWithParents_Should_ThrowException_When_ParentsAreInvalid()
    {
        Action act = () =>
            Neuron.CreateWithParents(_parents, _weights.SubVector(1, _weights.Count - 1), Bias, _activator);

        act.Should().Throw<ArgumentException>().WithMessage("Neuron parents and weights must be the same length");
    }

    [Test]
    public void SetParents_Should_CorrectlySetParents_When_ArgumentsAreValid()
    {
        var neuron = Neuron.Create(_weights.SubVector(0, _parents.Count), Bias, _activator);
        neuron.SetParents(_parents);
        neuron.Inputs.Should().BeNull();
        neuron.Parents.Should().BeEquivalentTo(_parents);
    }

    [Test]
    public void SetParents_Should_ThrowException_When_InputsAreSet()
    {
        var neuron = Neuron.CreateWithInputs(_inputs, _weights, Bias, _activator);

        var act = () => neuron.SetParents(_parents);

        act.Should().Throw<ArgumentException>().WithMessage("Cannot set neuron parents when inputs are already set");
    }

    [Test]
    public void SetParents_Should_ThrowException_When_ParentsAndWeightsAreNotTheSameLength()
    {
        var neuron = Neuron.Create(_weights, Bias, _activator);

        var act = () => neuron.SetParents(_parents);

        act.Should().Throw<ArgumentException>().WithMessage("Neuron parents and weights must be the same length");
    }

    [Test]
    public void SetInputs_Should_CorrectlySetInputs_When_ArgumentsAreValid()
    {
        var neuron = Neuron.Create(_weights, Bias, _activator);
        neuron.SetInputs(_inputs);
        neuron.Inputs.Should().BeEquivalentTo(_inputs);
        neuron.Parents.Should().BeNull();
    }

    [Test]
    public void SetInputs_Should_ThrowException_When_ParentsAreSet()
    {
        var neuron = Neuron.CreateWithParents(_parents, _weights.SubVector(0, _parents.Count), Bias, _activator);

        var act = () => neuron.SetInputs(_inputs);

        act.Should().Throw<ArgumentException>().WithMessage("Cannot set neuron inputs when parents are already set");
    }

    [Test]
    public void SetInputs_Should_ThrowException_When_InputsAndWeightsAreNotTheSameLength()
    {
        var neuron = Neuron.Create(_weights.SubVector(1, _weights.Count - 1), Bias, _activator);

        var act = () => neuron.SetInputs(_inputs);

        act.Should().Throw<ArgumentException>().WithMessage("Neuron inputs and weights must be the same length");
    }

    [Test]
    public void NeuronWithInputs_Should_ActivateCorrectly()
    {
        var neuron = Neuron.CreateWithInputs(_inputs, _weights, Bias, _activator);
        var output = neuron.Output;

        var expected = Bias;
        for (var i = 0; i < _inputs.Count; i++) expected += _inputs[i] * _weights[i];
        expected = _activator.Activate(expected);

        output.Should().Be(expected);
    }

    [Test]
    public void NeuronWithParents_Should_ActivateCorrectly()
    {
        var weights = _weights.SubVector(0, _parents.Count);
        var neuron = Neuron.CreateWithParents(_parents, weights, Bias, _activator);
        var output = neuron.Output;

        var expected = Bias;
        for (var i = 0; i < _parents.Count; i++) expected += _parents[i].Output * weights[i];
        expected = _activator.Activate(expected);

        output.Should().Be(expected);
    }

    [Test]
    public void NeuronWithoutParentsOrInput_Should_NotActivate()
    {
        var neuron = Neuron.Create(_weights, Bias, _activator);
        var act = () =>
        {
            var _ = neuron.Output;
        };

        act.Should().Throw<ArgumentNullException>();
    }
}