#include "MLGamesPolicyRunner.h"

void UMLGamesPolicyRunner::SetBackend(FMLGamesBackendDelegate InBackend)
{
	Backend = InBackend;
}

TArray<float> UMLGamesPolicyRunner::Predict(const TArray<float>& Observation) const
{
	check(Observation.Num() == ObservationSize);
	check(Backend.IsBound());
	TArray<float> Action = Backend.Execute(Observation);
	check(Action.Num() == ActionSize);
	return Action;
}
