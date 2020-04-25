#pragma once

#include <IPlayer.h>

namespace Player
{
	class BluPigPlayer : public IPlayer
	{
	public:
		virtual ~BluPigPlayer() {};

		int MakeMove(std::shared_ptr<GomokuGame> pGame, bool bTurn, int& moveToSave);

		void MoveMadeInGame(int /*moveIndex*/) {}
	};
}