#include "pch.h"

#include <BluPigPlayer.h>
#include <GomokuUtils.h>

#include <api/renju_api.h>
#include <ai/eval.h>

#include <iostream>

namespace Player
{
	int BluPigPlayer::MakeMove(std::shared_ptr<GomokuGame> pGame, bool bTurn, int& moveToSave)
	{
		if (pGame->GetMovesPlayed() == 0)
		{
			moveToSave = 112;
			return 112;
		}
		int player = bTurn ? 1 : 2;
		int move_r, move_c, winning_player, actual_depth;
		unsigned int node_count, eval_count;

		bool success = RenjuAPI::generateMove(pGame->GetBoard(), player, -1, 1500, 1, &actual_depth, &move_r, &move_c,
			&winning_player, &node_count, &eval_count, nullptr, nullptr);

		if (!success || move_r < 0 || move_c < 0)
		{
			std::cout << "RENJU API ERROR!";
			exit(10);
		}

		moveToSave = ConvertToIndex(move_r, move_c, 15);
		return moveToSave;
	}
}