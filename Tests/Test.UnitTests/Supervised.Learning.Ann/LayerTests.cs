using Common.Maths.ActivationFunction;
using Common.Maths.ActivationFunction.Interface;
using FluentAssertions;
using Learning.Supervised.Ann.Structure;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using static Common.Maths.ActivationFunction.Interface.IActivationFunction;

namespace Tests.Supervised.Learning.Ann;

// TODO: More tests

[TestFixture]
public class LayerTests
{
    private static readonly MatrixBuilder<double> M = Matrix<double>.Build;
    private static readonly VectorBuilder<double> V = Vector<double>.Build;
    private static readonly IActivationFunction ActivationFunction = new LinearActivator();

    private const int NumberOfNeurons = 10;
    private const int NumberOfInputs = 2;
    private const double MinWeight = -0.5;
    private const double MaxWeight = 0.75;

    [TestCaseSource(nameof(LayerDataSources))]
    public void Create_Should_CreateCorrectly(
        double[,] weightInputs,
        int numberOfNeurons,
        int numberOfInputs
    )
    {
        var weightsMatrix = M.DenseOfArray(weightInputs);
        var layer = Layer.Create(weightsMatrix, ActivationFunction);
        layer.InputWeights.RowCount.Should().Be(numberOfNeurons);
        layer.InputWeights.ColumnCount.Should().Be(numberOfInputs + 1);
        layer.InputWeights.Should().BeEquivalentTo(M.DenseOfArray(weightInputs));
        layer.ActivationFunction.Should().Be(ActivationFunction);
    }

    [TestCase(NumberOfNeurons)]
    [TestCase(100)]
    [TestCase(1)]
    public void CreateWithRandomWeights_Should_CreateWithRandomWeightsCorrectly(int numberOfNeurons)
    {
        var layer = Layer.CreateWithRandomWeights(
            numberOfNeurons,
            NumberOfInputs,
            ActivationFunction,
            MaxWeight,
            MinWeight
        );

        layer.InputWeights.RowCount.Should().Be(numberOfNeurons);
        layer.InputWeights.ColumnCount.Should().Be(NumberOfInputs + 1);
        layer.NumberOfNeurons.Should().Be(numberOfNeurons);

        layer.ActivationFunction.Should().Be(ActivationFunction);
    }

    [TestCase(0)]
    [TestCase(-10)]
    public void CreateWithRandomWeights_Should_ThrowException_When_TooFewNeurons(
        int numberOfNeurons
    )
    {
        Action act = () =>
            Layer.CreateWithRandomWeights(numberOfNeurons, NumberOfInputs, ActivationFunction);

        act.Should().Throw<ArgumentException>().WithMessage("Layer must be created with neurons");
    }

    [TestCase(0)]
    [TestCase(-10)]
    public void CreateWithRandomWeights_Should_ThrowException_When_TooFewInputs(int numberOfInputs)
    {
        Action act = () =>
            Layer.CreateWithRandomWeights(NumberOfNeurons, numberOfInputs, ActivationFunction);

        act.Should().Throw<ArgumentException>().WithMessage("Layer must be created with inputs");
    }

    [TestCase(0, 0)]
    [TestCase(10, 1)]
    public void CreateWithRandomWeights_Should_ThrowException_When_TooFewInputs(
        int minWeight,
        int maxWeight
    )
    {
        Action act = () =>
            Layer.CreateWithRandomWeights(
                NumberOfNeurons,
                NumberOfInputs,
                ActivationFunction,
                maxWeight,
                minWeight
            );

        act.Should()
            .Throw<ArgumentException>()
            .WithMessage("Minimum weight must be greater than maximum weight");
    }

    [Test]
    public void SetInputLayer_Should_CorrectlySetInputLayer()
    {
        var layerOne = Layer.CreateWithRandomWeights(
            NumberOfNeurons,
            NumberOfInputs,
            ActivationFunction
        );
        var layerTwo = Layer.CreateWithRandomWeights(
            NumberOfNeurons,
            NumberOfInputs,
            ActivationFunction
        );

        layerTwo.SetInputLayer(layerOne);

        layerTwo.InputLayer.Should().BeEquivalentTo(layerOne);
    }

    [Test]
    public void SetOutputLayer_Should_CorrectlySetOutputLayer()
    {
        var layerOne = Layer.CreateWithRandomWeights(
            NumberOfNeurons,
            NumberOfInputs,
            ActivationFunction
        );
        var layerTwo = Layer.CreateWithRandomWeights(
            NumberOfNeurons,
            NumberOfInputs,
            ActivationFunction
        );

        layerOne.SetOutputLayer(layerTwo);

        layerOne.OutputLayer.Should().BeEquivalentTo(layerTwo);
        layerOne.OutputWeights.Should().BeEquivalentTo(layerTwo.InputWeights);
    }

    [Test]
    public void ActivateShould_CorrectlyActivateLayer_WithGivenInputs_And_WithPreviousOutputs()
    {
        var layerOne = Layer.Create(
            M.DenseOfArray(
                new[,]
                {
                    { 1.0, 2.0, 3.0 },
                    { 4.0, 5.0, 6.0 },
                }
            ),
            new LinearActivator()
        );
        var layerTwo = Layer.Create(
            M.DenseOfArray(
                new[,]
                {
                    { 7.0, 8.0, 9.0 },
                }
            ),
            new LinearActivator()
        );

        layerTwo.SetInputLayer(layerOne);
        layerOne.SetOutputLayer(layerTwo);

        layerOne.Activate(V.DenseOfArray([1, 2]));
        layerTwo.Activate();

        layerOne.Outputs.Should().NotBeNull();
        layerOne.Outputs.Should().BeEquivalentTo(V.DenseOfArray([8, 20]));

        layerTwo.Outputs.Should().NotBeNull();
        layerTwo.Outputs.Should().BeEquivalentTo(V.DenseOfArray([225]));
    }

    private static readonly object[] LayerDataSources =
    [
        new object[]
        {
            new double[,]
            {
                { 1, 2, 3 },
            },
            1,
            2,
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
            2,
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
            3,
        },
    ];
}
