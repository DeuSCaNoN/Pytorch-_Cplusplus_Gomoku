#pragma once

#include <IPlayer.h>

namespace Player
{
	class BluPigPlayer : public IPlayer
	{
	public:
		virtual ~BluPigPlayer() {};

		int MakeMove(std::shared_ptr<GomokuGame> pGame, bool bTurn, float* moveEstimates);

		std::string PrintWinningStatement();

		void MoveMadeInGame(int /*moveIndex*/) {}

		void ClearTree() {}
	};
}