using Learning.Supervised.Training.Algorithm.Interface;
using Learning.Supervised.Training.Data;
using Learning.Supervised.Training.LearningRate.Interface;
using Learning.Supervised.Training.LossFunction.Interface;

namespace Learning.Supervised.Ann.Algorithm;

public class BackPropagationWithGradientDescent : ITrainer
{
    private readonly Ann _ann;

    public BackPropagationWithGradientDescent(
        ILearningRate learningRate,
        ILossFunction lossFunction,
        SupervisedLearningData data,
        Ann ann
    )
    {
        LearningRate = learningRate;
        LossFunction = lossFunction;
        Data = data;
        _ann = ann;
    }

    public SupervisedLearningData Data { get; private init; }
    public ILearningRate LearningRate { get; private init; }
    public ILossFunction LossFunction { get; private init; }

    public void Train()
    {
        try
        {
            if (!_ann.HasBeenBuilt)
                throw new InvalidOperationException("Cannot train Ann before building.");

            for (var epoch = 0; epoch < Data.MaxEpochs; epoch++)
                TrainOnce(epoch);
        }
        catch (Exception e)
        {
            throw new Exception("Error running the Ann during gradient descent: ", e);
        }
    }

    private void TrainOnce(int epoch)
    {
        var (inputs, expectedOutputs) = Data.GetInputsOutputs(epoch);

        if (!_ann.HasRun)
            _ann.Run(inputs);

        var outputs = _ann.Outputs;

        var loss = LossFunction.CalculateLoss(expectedOutputs, outputs);
    }
}
