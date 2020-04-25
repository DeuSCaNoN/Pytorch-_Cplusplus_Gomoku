#pragma once

#include <GomokuGame.h>

#include <memory>
namespace Player
{
	class IPlayer
	{
	public:
		virtual ~IPlayer() = default;

		virtual int MakeMove(std::shared_ptr<GomokuGame> pGame, bool bTurn, int& moveToSave) = 0;

		virtual void MoveMadeInGame(int moveIndex) = 0;
	};
}