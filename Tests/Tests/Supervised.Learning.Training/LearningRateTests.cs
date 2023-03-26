using FluentAssertions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Training.LearningRate;

namespace Tests.Supervised.Learning.Training
{
    [TestFixture]
    public class LearningRateTests
    {
        [TestCase(0.01, 1)]
        [TestCase(0.1, 20)]
        [TestCase(0.0, 1.234)]
        [TestCase(5.6, 1023)]
        public void FlatLearningRate_Should_ApplyCorrentLearningRate(double learningRate, double errorSignal)
        {
            var result = new FlatLearningRate(learningRate).ApplyLearningRate(errorSignal);
            result.Should().Be(learningRate * errorSignal);
        }
    }
}
