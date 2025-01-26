using Common.Maths.ActivationFunction;
using FluentAssertions;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace Tests.Common.Maths;

[TestFixture]
public class ActivatorTests
{
    private static readonly VectorBuilder<double> V = Vector<double>.Build;

    [TestCase(0, 0)]
    [TestCase(-1, 0)]
    [TestCase(1, 1)]
    [TestCase(999, 1)]
    public void BinaryActivator_Should_Return_CorrectValues(double input, double expectedOutput)
    {
        var activator = new BinaryActivator();
        var (output, derivative) = activator.Activate(input);
        output.Should().Be(expectedOutput);
        derivative.Should().Be(0);
    }

    [Test]
    public void VectorBinaryActivator_Should_Return_CorrectValues()
    {
        var activator = new BinaryActivator();
        var (outputs, derivatives) = activator.Activate(
            V.DenseOfArray([-999, -1, -0.5, 0.0, 0.4, 0.6, 1, 999])
        );
        outputs.Should().BeEquivalentTo(V.DenseOfArray([0, 0, 0, 0, 1, 1, 1, 1]));
        derivatives.Should().BeEquivalentTo(V.DenseOfArray([0, 0, 0, 0, 0, 0, 0, 0]));
    }

    [TestCase(0, 0)]
    [TestCase(-1, -1)]
    [TestCase(1, 1)]
    [TestCase(999, 999)]
    public void LinearActivator_Should_Return_CorrectValues(double input, double expectedOutput)
    {
        var activator = new LinearActivator();
        var (output, derivative) = activator.Activate(input);
        output.Should().Be(expectedOutput);
        derivative.Should().Be(1);
    }

    [Test]
    public void VectorLinearActivator_Should_Return_CorrectValues()
    {
        var inputs = V.DenseOfArray([-999, -1, -0.5, 0.0, 0.5, 1, 999]);
        var activator = new LinearActivator();
        var (outputs, derivatives) = activator.Activate(inputs);
        outputs.Should().BeEquivalentTo(inputs);
        derivatives.Should().BeEquivalentTo(V.DenseOfArray([1, 1, 1, 1, 1, 1, 1]));
    }

    [TestCase(0, 0.5, 0.25)]
    [TestCase(0.458, 0.61253961, 0.23733483)]
    [TestCase(1.6, 0.83201838, 0.13976379)]
    [TestCase(-3, 0.04742587, 0.04517665)]
    public void SigmoidActivator_Should_Return_CorrectValues(
        double input,
        double expectedOutput,
        double expectedDerivative
    )
    {
        var activator = new SigmoidActivator();
        var (output, derivative) = activator.Activate(input);
        output.Should().BeApproximately(expectedOutput, 0.00000001);
        derivative.Should().BeApproximately(expectedDerivative, 0.00000001);
    }

    [Test]
    public void VectorSigmoidActivator_Should_Return_CorrectValues()
    {
        var activator = new SigmoidActivator();
        var (outputs, derivatives) = activator.Activate(V.DenseOfArray([0, 0.458, 1.6, -3]));
        outputs
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([0.5, 0.61253961, 0.83201838, 0.04742587]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
        derivatives
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([0.25, 0.23733483, 0.13976379, 0.04517665]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
    }

    [TestCase(0, 0, 1)]
    [TestCase(2, 0.96402758, 0.070650829)]
    [TestCase(-1, -0.76159415, 0.41997434)]
    [TestCase(-0.22, -0.21651806, 0.95311992)]
    public void TanhActivator_Should_Return_CorrectValues(
        double input,
        double expectedOutput,
        double expectedDerivative
    )
    {
        var activator = new TanhActivator();
        var (output, derivative) = activator.Activate(input);
        output.Should().BeApproximately(expectedOutput, 0.00000001);
        derivative.Should().BeApproximately(expectedDerivative, 0.00000001);
    }

    [Test]
    public void VectorTanhActivator_Should_Return_CorrectValues()
    {
        var activator = new TanhActivator();
        var (outputs, derivatives) = activator.Activate(V.DenseOfArray([0, 2, -1, -0.22]));
        outputs
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([0, 0.964027581, -0.76159415, -0.21651806]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
        derivatives
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([1, 0.070650829, 0.41997434, 0.95311992]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
    }

    [TestCase(0, 0)]
    [TestCase(-1, 0)]
    [TestCase(1, 1)]
    [TestCase(999, 999)]
    public void ReLuActivator_Should_Return_CorrectValues(double input, double expectedOutput)
    {
        var activator = new ReLuActivator();
        var (output, derivative) = activator.Activate(input);
        output.Should().Be(expectedOutput);
        derivative.Should().Be(input >= 0 ? 1 : 0);
    }

    [Test]
    public void VectorReLuActivator_Should_Return_CorrectValues()
    {
        var activator = new ReLuActivator();
        var (outputs, derivatives) = activator.Activate(V.DenseOfArray([0, -1, 1, 999]));
        outputs
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([0, 0, 1, 999]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
        derivatives
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([1, 0, 1, 1]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
    }

    [TestCase(0, 0, 0.1)]
    [TestCase(-1, -0.1, 0.1)]
    [TestCase(-100, -10, 0.1)]
    [TestCase(-100, -50, 0.5)]
    [TestCase(999, 999, 0.1)]
    public void LeakyReLuActivator_Should_Return_CorrectValues(
        double input,
        double expectedOutput,
        double leak
    )
    {
        var activator = LeakyReLuActivator.Create(leak);
        var (output, derivative) = activator.Activate(input);
        output.Should().Be(expectedOutput);
        derivative.Should().Be(output >= 0 ? 1 : leak);
    }

    [Test]
    public void VectorLeakyReLuActivator_Should_Return_CorrectValues()
    {
        const double leak = 0.1;
        var activator = LeakyReLuActivator.Create(leak);
        var (outputs, derivatives) = activator.Activate(V.DenseOfArray([0, -1, -100, 999]));
        outputs
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([0, -1.0 * leak, -100 * leak, 999]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
        derivatives
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([1, leak, leak, 1]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
    }

    [Test]
    public void LeakyReLuActivator_Should_Throw_ArgumentOutOfRangeException_When_LeakIsNegative()
    {
        Action act = () => _ = LeakyReLuActivator.Create(-0.1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestCase(10, 10, 1.0, 0.1)]
    [TestCase(0.5, 0.5, 1.0, 0.1)]
    [TestCase(0, 0, 1.0, 0.1)]
    [TestCase(-10, -0.09999546, 0.00000453, 0.1)]
    [TestCase(-10, -0.49997730, 0.00002269, 0.5)]
    [TestCase(-100, -0.1, 0, 0.1)]
    public void ELuActivator_Should_Return_CorrectValues(
        double input,
        double expectedOutput,
        double expectedDerivative,
        double alpha
    )
    {
        var activator = ELuActivator.Create(alpha);
        var (output, derivative) = activator.Activate(input);
        output.Should().BeApproximately(expectedOutput, 0.00000001);
        derivative.Should().BeApproximately(expectedDerivative, 0.00000001);

        Console.WriteLine(derivative);
    }

    [Test]
    public void VectorELuActivator_Should_Return_CorrectValues()
    {
        const double alpha = 0.1;
        var activator = ELuActivator.Create(alpha);
        var (outputs, derivatives) = activator.Activate(V.DenseOfArray([10, 0.5, 0, -10, -100]));
        outputs
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([10, 0.5, 0, -0.09999546, -0.1]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
        derivatives
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([1.0, 1.0, 1.0, 0.00000453, 0.0]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
    }

    [Test]
    public void ELuActivator_Should_Throw_ArgumentOutOfRangeException_When_AlphaIsNegative()
    {
        Action act = () => _ = ELuActivator.Create(-0.1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestCase(0, 0, 0.5, 1)]
    [TestCase(0.458, 0.28054314, 0.72123896, 1)]
    [TestCase(1.6, 1.33122941, 1.05564045, 1)]
    [TestCase(1.6, 0.92691880, 0.65731070, 0.2)]
    [TestCase(-3, -0.14227761, -0.088104101, 1)]
    public void SwishActivator_Should_Return_CorrectValues(
        double input,
        double expectedOutput,
        double expectedDerivative,
        double b
    )
    {
        var activator = SwishActivator.Create(b);
        var (output, derivative) = activator.Activate(input);
        output.Should().BeApproximately(expectedOutput, 0.00000001);
        derivative.Should().BeApproximately(expectedDerivative, 0.00000001);
    }

    [Test]
    public void VectorLSwishActivator_Should_Return_CorrectValues_For_OneBeta()
    {
        var activator = SwishActivator.Create(1);
        var (outputs, derivatives) = activator.Activate(V.DenseOfArray([0, 0.458, 1.6, -3]));
        outputs
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([0, 0.28054314, 1.33122941, -0.14227761]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
        derivatives
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([0.5, 0.72123896, 1.05564045, -0.08810410]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
    }

    [Test]
    public void VectorLSwishActivator_Should_Return_CorrectValues_For_NonOneBeta()
    {
        var activator = SwishActivator.Create(0.2);
        var (outputs, derivatives) = activator.Activate(V.DenseOfArray([1.6]));
        outputs
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([0.92691880]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
        derivatives
            .Should()
            .BeEquivalentTo(
                V.DenseOfArray([0.65731070]),
                options =>
                    options
                        .Using<double>(ctx =>
                            ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.00000001)
                        )
                        .WhenTypeIs<double>()
            );
    }

    [Test]
    public void SwishActivator_Should_Throw_ArgumentOutOfRangeException_When_BIsNegative()
    {
        Action act = () => _ = SwishActivator.Create(-0.1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
