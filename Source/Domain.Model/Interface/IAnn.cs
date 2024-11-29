using Learning.Supervised.Training.Algorithm.Interface;

namespace Learning.Supervised.Ann.Interface;

public interface IAnn : ILearner
{
    public void Run();
}