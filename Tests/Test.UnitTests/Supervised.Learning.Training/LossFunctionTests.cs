using FluentAssertions;
using Learning.Supervised.Training.LossFunction;

namespace Tests.Supervised.Learning.Training
{
    [TestFixture]
    public class LossFunctionTests
    {
        [Test]
        public void MeanSquaredError_Should_ThrowException_When_ExpectedIsNull()
        {
            Action act = () => new MeanSquaredError().CalculateLoss(null!, Array.Empty<double>());

            act.Should().Throw<ArgumentNullException>();
        }

        [Test]
        public void MeanSquaredError_Should_ThrowException_When_ActualIsNull()
        {
            Action act = () => new MeanSquaredError().CalculateLoss(Array.Empty<double>(), null!);

            act.Should().Throw<ArgumentNullException>();
        }

        [Test]
        public void MeanSquaredError_Should_ThrowException_When_ActualAndExpectedAreDifferentLengths()
        {
            Action act = () => new MeanSquaredError().CalculateLoss(Array.Empty<double>(), new double[] { 1 });

            act.Should().Throw<ArgumentException>().WithMessage("Expected and actual should be the same length.");
        }

        [Test]
        public void MeanSquaredError_Should_CalculateMSEWithMultipleInputs()
        {
            var expected = new double[] { 1, 2, 3 };
            var actual = new double[] { 2, 4, 6 };

            var mse = new MeanSquaredError().CalculateLoss(expected, actual);
            var expectedOutput = Math.Sqrt(Math.Pow(2 - 1, 2)) + Math.Sqrt(Math.Pow(4 - 2, 2)) + Math.Sqrt(Math.Pow(6 - 3, 2));
            expectedOutput /= 3;

            mse.Should().Be(expectedOutput);
        }

        [Test]
        public void MeanSquaredError_Should_CalculateMSEWithSingleInputs()
        {
            var mse = new MeanSquaredError().CalculateLoss(2, 1);
            var expectedOutput = Math.Sqrt(Math.Pow(2 - 1, 2));

            mse.Should().Be(expectedOutput);
        }
    }
}
