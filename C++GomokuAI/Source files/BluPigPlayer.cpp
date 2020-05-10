#include "pch.h"

#include <BluPigPlayer.h>
#include <GomokuUtils.h>
#include <FwdDecl.h>

#include <api/renju_api.h>

#include <iostream>

namespace Player
{
	int BluPigPlayer::MakeMove(std::shared_ptr<GomokuGame> pGame, bool bTurn, float* pMoveEstimates)
	{
		memset(pMoveEstimates, 0, BOARD_LENGTH);
		if (pGame->GetMovesPlayed() == 0)
		{
			pMoveEstimates[112] = 1.0f;
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

		int moveMade = ConvertToIndex(move_r, move_c, 15);
		pMoveEstimates[moveMade] = 1.0f;
		return moveMade;
	}
}