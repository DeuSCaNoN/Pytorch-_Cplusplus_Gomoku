#pragma once
#pragma once

#include <IPlayer.h>

namespace Player
{
	class HumanPlayer : public IPlayer
	{
	public:
		virtual ~HumanPlayer() {};

		int MakeMove(std::shared_ptr<GomokuGame> pGame, bool bTurn, float* moveEstimates);

		std::string PrintWinningStatement();

		void MoveMadeInGame(int /*moveIndex*/) {}

		void ClearTree() {}
	};
}