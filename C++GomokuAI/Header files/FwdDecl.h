#pragma once

#define BOARD_LENGTH 225
#define BOARD_SIDE 15
#define BOARD_WIN 5

class GomokuGame;

namespace LibTorchAgent
{
	class GomokuPolicyAgent;
}

namespace torch
{
	class Tensor;
	
	namespace nn
	{
		class Module;

		class Conv2d;
		class Linear;
		class BatchNorm2d;
	}
}

typedef std::function<void(torch::Tensor const&, double)>* pPredictCbFnPtr_t;

namespace MonteCarlo
{
	class MonteCarloTreeSearch;
	class MonteCarloNode;
}