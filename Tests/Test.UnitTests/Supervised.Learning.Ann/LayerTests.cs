using Common.Maths.ActivationFunction;
using Common.Maths.ActivationFunction.Interface;
using FluentAssertions;
using Learning.Supervised.Ann.Structure;
using MathNet.Numerics.LinearAlgebra;
using static Common.Maths.ActivationFunction.Interface.IActivationFunction;

namespace Tests.Supervised.Learning.Ann;

[TestFixture]
public class LayerTests
{
    private static readonly MatrixBuilder<double> _m = Matrix<double>.Build;
    private static readonly VectorBuilder<double> _v = Vector<double>.Build;
    private static readonly IActivationFunction _activator = new LinearActivator();
    private const ActivationFunction ActivationFunction = IActivationFunction
        .ActivationFunction
        .Linear;

    private const int NumberOfNeurons = 10;
    private const int NumberOfWeights = 20;
    private const double MinWeight = -0.5;
    private const double MaxWeight = 0.75;

    [TestCaseSource(nameof(_layerDataSources))]
    public void Create_Should_CreateCorrectly(
        double[,] weightInputs,
        int numberOfNeurons,
        int numberOfWeights
    )
    {
        var weightsMatrix = _m.DenseOfArray(weightInputs);
        var layer = Layer.Create(weightsMatrix, ActivationFunction);
        layer.Weights.Should().BeEquivalentTo(_m.DenseOfArray(weightInputs));
        layer.Activator.Should().Be(ActivationFunction);
    }

    [TestCase(NumberOfNeurons, NumberOfWeights, MinWeight, MaxWeight)]
    [TestCase(100, 200, 0, 1)]
    [TestCase(1, 1, 0, 10)]
    public void CreateWithRandomWeights_Should_CreateWithRandomWeightsCorrectly(
        int numberOfNeurons,
        int numberOfWeights,
        double minWeight,
        double maxWeight
    )
    {
        var layer = Layer.CreateWithRandomWeights(
            numberOfNeurons,
            numberOfWeights,
            minWeight,
            maxWeight,
            ActivationFunction
        );

        layer.BuildWeights();
        layer.Neurons.Should().HaveCount(numberOfNeurons);
        foreach (var neuron in layer.Neurons)
        {
            neuron.Weights.Should().HaveCount(numberOfWeights);
            neuron.Bias.Should().BeInRange(minWeight, maxWeight);
            foreach (var weight in neuron.Weights)
                weight.Should().BeInRange(minWeight, maxWeight);
        }

        layer.Activator.Should().Be(ActivationFunction);
    }

    [TestCase(0)]
    [TestCase(-10)]
    public void CreateWithRandomWeights_Should_ThrowException_When_TooFewNeurons(
        int numberOfNeurons
    )
    {
        Action act = () =>
            Layer.CreateWithRandomWeights(
                numberOfNeurons,
                NumberOfWeights,
                MinWeight,
                MaxWeight,
                ActivationFunction
            );

        act.Should().Throw<ArgumentException>().WithMessage("Layer must be created with neurons");
    }

    [TestCase(0)]
    [TestCase(-10)]
    public void CreateWithRandomWeights_Should_ThrowException_When_TooFewWeights(
        int numberOfWeights
    )
    {
        Action act = () =>
            Layer.CreateWithRandomWeights(
                NumberOfNeurons,
                numberOfWeights,
                MinWeight,
                MaxWeight,
                ActivationFunction
            );

        act.Should().Throw<ArgumentException>().WithMessage("Layer must be created with weights");
    }

    [TestCase(5, 0)]
    [TestCase(0.1, -0.1)]
    public void CreateWithRandomWeights_Should_ThrowException_When_MinWeightLessThanMaxWeight(
        double minWeight,
        double maxWeight
    )
    {
        Action act = () =>
            Layer.CreateWithRandomWeights(
                NumberOfNeurons,
                NumberOfWeights,
                minWeight,
                maxWeight,
                ActivationFunction
            );

        act.Should()
            .Throw<ArgumentException>()
            .WithMessage("Min weight must be less than max weight");
    }

    [TestCaseSource(nameof(_layerDataSources))]
    public void BuildWeights_Should_CorrectlyBuildNeuronWeights(
        double[,] weightInputs,
        int numberOfNeurons,
        int numberOfWeights
    )
    {
        var weightsMatrix = _m.DenseOfArray(weightInputs);
        var layer = Layer.Create(weightsMatrix, ActivationFunction).BuildWeights();

        layer.Neurons.Should().HaveCount(numberOfNeurons);
        var weights = weightInputs.Cast<double>().ToList();
        for (var i = 0; i < numberOfNeurons; i++)
        {
            var neuronWeights = weights.GetRange(i * numberOfWeights, numberOfWeights);
            layer
                .Neurons[i]
                .Weights.Should()
                .BeEquivalentTo(neuronWeights.GetRange(1, neuronWeights.Count - 1));
            layer.Neurons[i].Bias.Should().Be(neuronWeights[0]);
            layer.Neurons[i].Activator.Should().BeEquivalentTo(_activator);
            layer.Neurons[i].Parents.Should().BeNull();
            layer.Neurons[i].Inputs.Should().BeNull();
        }
    }

    [Test]
    public void Create_Should_ThrowException_When_WeightsAreEmpty()
    {
        Action act = () => Layer.Create(_m.DenseOfArray(new double[,] { }), ActivationFunction);

        act.Should().Throw<ArgumentException>().WithMessage("Layer must be created with weights");
    }

    private static object[] _layerDataSources =
    [
        new object[]
        {
            new double[,]
            {
                { 1, 2, 3 },
            },
            1,
            3,
        },
        new object[]
        {
            new double[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 },
            },
            3,
            3,
        },
        new object[]
        {
            new double[,]
            {
                { 1, 2, 3, 0 },
                { 4, 5, 6, 0 },
                { 7, 8, 9, 0 },
            },
            3,
            4,
        },
    ];

    [Test]
    public void AddInputs_Should_ThrowException_When_NoInputsProvided()
    {
        var layer = Layer
            .CreateWithRandomWeights(
                NumberOfNeurons,
                NumberOfWeights,
                MinWeight,
                MaxWeight,
                ActivationFunction
            )
            .BuildWeights();
        Action act = () => layer.SetInputs(_v.DenseOfArray(new double[] { }));

        act.Should().Throw<ArgumentException>().WithMessage("Must have at least one input");
    }

    [Test]
    public void AddInputs_Should_Succeed_When_InputsAreValid()
    {
        var inputs = new double[NumberOfNeurons];
        for (var i = 0; i < NumberOfNeurons; i++)
            inputs[i] = i;
        var layer = Layer
            .CreateWithRandomWeights(
                NumberOfNeurons,
                NumberOfNeurons,
                MinWeight,
                MaxWeight,
                ActivationFunction
            )
            .BuildWeights()
            .SetInputs(_v.DenseOfArray(inputs));

        foreach (var neuron in layer.Neurons)
            neuron.Inputs!.Should().BeEquivalentTo(inputs);
    }

    [Test]
    public void AddParentLayer_Should_Succeed_When_ValidParents()
    {
        var parentCount = 2;
        var parents = Layer
            .CreateWithRandomWeights(parentCount, parentCount, 0, 1, ActivationFunction)
            .BuildWeights();
        var layer = Layer
            .CreateWithRandomWeights(parentCount, parentCount, 0, 1, ActivationFunction)
            .BuildWeights()
            .AddParentLayer(parents);

        foreach (var neuron in layer.Neurons)
            neuron.Parents.Should().BeEquivalentTo(parents.Neurons);
    }

    [Test]
    public void Clone_Should_CloneTheLayer()
    {
        var layer = Layer
            .CreateWithRandomWeights(
                NumberOfNeurons,
                NumberOfWeights,
                MinWeight,
                MaxWeight,
                ActivationFunction
            )
            .BuildWeights();
        var clone = layer.Clone();

        clone.Weights.Should().BeEquivalentTo(layer.Weights);
        clone.Activator.Should().Be(layer.Activator);
        clone.Neurons.Should().BeEmpty();
        clone.IsBuilt.Should().BeFalse();
        clone.HasInputs.Should().BeFalse();
    }
}
