using Common.Maths.ActivationFunction;
using FluentAssertions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tests.Common.Maths
{
    [TestFixture]
    internal class ActivatorTests
    {
        [TestCase(0, 0)]
        [TestCase(-1, 0)]
        [TestCase(1, 1)]
        [TestCase(999, 1)]
        public void BinaryActivator_Should_Return_CorrectValues(double input, double output)
        {
            var result = new BinaryActivator().Activate(input);
            result.Should().Be(output);
        }

        [TestCase(0, 0)]
        [TestCase(-1, -1)]
        [TestCase(1, 1)]
        [TestCase(999, 999)]
        public void LinearActivator_Should_Return_CorrectValues(double input, double output)
        {
            var result = new LinearActivator().Activate(input);
            result.Should().Be(output);
        }

        [TestCase(0, 0.5)]
        [TestCase(0.458, 0.61253961)]
        [TestCase(1.6, 0.83201838)]
        [TestCase(-3, 0.04742587)]
        public void SigmoidActivator_Should_Return_CorrectValues(double input, double output)
        {
            var result = new SigmoidActivator().Activate(input);
            result.Should().BeApproximately(output, 0.00000001);
        }

        [TestCase(0, 0)]
        [TestCase(2, 0.96402758)]
        [TestCase(-1, -0.76159415)]
        [TestCase(-0.22, -0.21651806)]
        public void TanhActivator_Should_Return_CorrectValues(double input, double output)
        {
            var result = new TanhActivator().Activate(input);
            result.Should().BeApproximately(output, 0.00000001);
        }

        [TestCase(0, 0)]
        [TestCase(-1, 0)]
        [TestCase(1, 1)]
        [TestCase(999, 999)]
        public void ReLuActivator_Should_Return_CorrectValues(double input, double output)
        {
            var result = new ReLuActivator().Activate(input);
            result.Should().Be(output);
        }

        [TestCase(0, 0, 0.1)]
        [TestCase(-1, -0.1, 0.1)]
        [TestCase(-100, -10, 0.1)]
        [TestCase(-100, -50, 0.5)]
        [TestCase(999, 999, 0.1)]
        public void LeakyReLuActivator_Should_Return_CorrectValues(double input, double output, double leak)
        {
            var result = new LeakyReLuActivator(leak).Activate(input);
            result.Should().Be(output);
        }

        [Test]
        public void LeakyReLuActivator_Should_Throw_ArgumentOutOfRangeException_When_LeakIsNegative()
        {
            Action act = () => new LeakyReLuActivator(-0.1);

            act.Should().Throw<ArgumentOutOfRangeException>();
        }

        [TestCase(0, 0, 0.1)]
        [TestCase(-10, -0.09999546, 0.1)]
        [TestCase(-100, -0.1, 0.1)]
        [TestCase(-100, -0.5, 0.5)]
        [TestCase(999, 999, 0.1)]
        public void ELuActivator_Should_Return_CorrectValues(double input, double output, double alpha)
        {
            var result = new ELuActivator(alpha).Activate(input);
            result.Should().BeApproximately(output, 0.00000001);
        }

        [Test]
        public void ELuActivator_Should_Throw_ArgumentOutOfRangeException_When_AlphaIsNegative()
        {
            Action act = () => new ELuActivator(-0.1);

            act.Should().Throw<ArgumentOutOfRangeException>();
        }
    }
}
