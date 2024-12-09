using Common.Maths.ActivationFunction;
using Common.Maths.ActivationFunction.Interface;
using FluentAssertions;
using Learning.Supervised.Ann.Structure;
using MathNet.Numerics.LinearAlgebra;
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
    private const int NumberOfWeights = 20;
    private const double MinWeight = -0.5;
    private const double MaxWeight = 0.75;

    [TestCaseSource(nameof(LayerDataSources))]
    public void Create_Should_CreateCorrectly(
        double[,] weightInputs,
        int numberOfNeurons,
        int numberOfWeights
    )
    {
        var weightsMatrix = M.DenseOfArray(weightInputs);
        var layer = Layer.Create(weightsMatrix, ActivationFunction);
        layer.InputWeights.Should().BeEquivalentTo(M.DenseOfArray(weightInputs));
        layer.ActivationFunction.Should().Be(ActivationFunction);
    }

    [TestCase(NumberOfNeurons)]
    [TestCase(100)]
    [TestCase(1)]
    public void CreateWithRandomWeights_Should_CreateWithRandomWeightsCorrectly(int numberOfNeurons)
    {
        var layer = Layer.CreateWithRandomWeights(numberOfNeurons, ActivationFunction);

        // layer.InputWeights.RowCount.Should().Be(numberOfWeights);
        layer.NumberOfNeurons.Should().Be(numberOfNeurons);

        layer.ActivationFunction.Should().Be(ActivationFunction);
    }

    [TestCase(0)]
    [TestCase(-10)]
    public void CreateWithRandomWeights_Should_ThrowException_When_TooFewNeurons(
        int numberOfNeurons
    )
    {
        Action act = () => Layer.CreateWithRandomWeights(numberOfNeurons, ActivationFunction);

        act.Should().Throw<ArgumentException>().WithMessage("Layer must be created with neurons");
    }

    [Test]
    public void Create_Should_ThrowException_When_WeightsAreEmpty()
    {
        Action act = () => Layer.Create(M.DenseOfArray(Array), ActivationFunction);

        act.Should().Throw<ArgumentException>().WithMessage("Layer must be created with weights");
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
    private static readonly double[,] Array = new double[,] { };
}
