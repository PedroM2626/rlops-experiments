#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "MLGamesPolicyRunner.generated.h"

DECLARE_DELEGATE_RetVal_OneParam(TArray<float>, FMLGamesBackendDelegate, const TArray<float>&);

UCLASS(BlueprintType)
class UMLGamesPolicyRunner : public UObject
{
	GENERATED_BODY()

public:
	static constexpr int32 ObservationSize = 10;
	static constexpr int32 ActionSize = 2;

	void SetBackend(FMLGamesBackendDelegate InBackend);
	TArray<float> Predict(const TArray<float>& Observation) const;

private:
	FMLGamesBackendDelegate Backend;
};
