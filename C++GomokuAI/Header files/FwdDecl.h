#pragma once

#define BOARD_LENGTH 225
#define BOARD_SIDE 15
#define BOARD_WIN 5

class GomokuGame;

namespace LibTorchAgent
{
	class GomokuPolicyAgent;
}

namespace MonteCarlo
{
	class MonteCarloTreeSearch;
	class MonteCarloNode;
}