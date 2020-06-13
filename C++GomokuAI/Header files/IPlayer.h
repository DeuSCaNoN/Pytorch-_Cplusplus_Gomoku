#pragma once

#include <GomokuGame.h>

#include <memory>
namespace Player
{
	class IPlayer
	{
	public:
		virtual ~IPlayer() = default;

		virtual int MakeMove(std::shared_ptr<GomokuGame> pGame, bool bTurn, float* moveEstimates) = 0;

		virtual std::string PrintWinningStatement() = 0;

		virtual void ClearTree() = 0;
	};
}