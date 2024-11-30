using FluentAssertions;
using Learning.Supervised.Training.LossFunction;
using MathNet.Numerics.LinearAlgebra;

namespace Tests.Supervised.Learning.Training;

[TestFixture]
public class LossFunctionTests
{
    [Test]
    public void MeanSquaredError_Should_ThrowException_When_ActualAndExpectedAreDifferentLengths()
    {
        Action act = () =>
            new MeanSquaredError().CalculateLoss(
                Vector<double>.Build.Random(0),
                Vector<double>.Build.Random(1)
            );

        act.Should()
            .Throw<ArgumentException>()
            .WithMessage("Expected and actual should be the same length.");
    }

    [Test]
    public void MeanSquaredError_Should_CalculateMSEWithMultipleInputs()
    {
        var expected = Vector<double>.Build.DenseOfArray([1, 2, 3]);
        var actual = Vector<double>.Build.DenseOfArray([2, 4, 6]);

        var mse = new MeanSquaredError().CalculateLoss(expected, actual);
        var expectedOutput = Math.Pow(2 - 1, 2) + Math.Pow(4 - 2, 2) + Math.Pow(6 - 3, 2);
        expectedOutput /= 3;

        mse.Should().Be(expectedOutput);
    }

    [Test]
    public void MeanSquaredError_Should_CalculateMSEWithSingleInputs()
    {
        var mse = new MeanSquaredError().CalculateLoss(2, 1);
        var expectedOutput = Math.Pow(2 - 1, 2);

        mse.Should().Be(expectedOutput);
    }
}
