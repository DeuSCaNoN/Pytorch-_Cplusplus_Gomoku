#include "pch.h"

#include <HumanPlayer.h>
#include <GomokuUtils.h>
#include <FwdDecl.h>

namespace Player
{
	int HumanPlayer::MakeMove(std::shared_ptr<GomokuGame> /*pGame*/, bool /*bTurn*/, float* pMoveEstimates)
	{
		int row, col;
		std::cout << "Row: " << std::endl;
		std::cin >> row;
		std::cout << "Col: " << std::endl;
		std::cin >> col;
		int index = ConvertToIndex(row, col, BOARD_SIDE);

		if (pMoveEstimates)
		{
			memset(pMoveEstimates, 0, BOARD_LENGTH * sizeof(float));
			pMoveEstimates[index] = 1.0f;
		}
		return index;
	}

	std::string HumanPlayer::PrintWinningStatement()
	{
		return "Human won";
	}
}