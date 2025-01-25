using FluentAssertions;
using Learning.Supervised.Training.LearningRate;

namespace Tests.Supervised.Learning.Training;

[TestFixture]
public class LearningRateTests
{
    [TestCase(0.01, 1)]
    [TestCase(0.1, 20)]
    [TestCase(0.0, 1.234)]
    [TestCase(5.6, 1023)]
    public void FlatLearningRate_Should_ApplyCorrectLearningRate(
        double learningRate,
        double errorSignal
    )
    {
        var result = new FlatLearningRate(learningRate).Apply(errorSignal);
        result.Should().Be(learningRate * errorSignal);
    }
}
