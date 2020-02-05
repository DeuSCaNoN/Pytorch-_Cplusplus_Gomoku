#include "pch.h"
#include "GomokuUtils.h"

#include "GomokuGame.h"

#include <api/renju_api.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>

namespace GomokuUtils
{
	void ConvertToRowCol(
		int index,
		int sideLength,
		short& row, /*out*/
		short& col /*out*/)
	{
		row = index / sideLength;
		col = index % sideLength;
	}

	void GenerateDataSet(short boardSide)
	{
		GomokuGame game = GomokuGame(boardSide, 5);

		std::ofstream myFile;
		myFile.open("dataset.txt", std::ofstream::binary);

		for (int i = 0; i < 15; i++)
		{
			srand((unsigned int)time(NULL));
			int initialMove = rand() % (boardSide * boardSide);
			game.PlayMove(initialMove);
			int lastMove = initialMove;

			while (game.GetGameWinState() == WinnerState_enum::None)
			{
				myFile.write(game.GetBoard(), boardSide * boardSide);
				myFile << std::endl;
				int move_r, move_c, winning_player, actual_depth;
				unsigned int node_count, eval_count;

				bool success = RenjuAPI::generateMove(game.GetBoard(), 1, -1, 1500, 1, &actual_depth, &move_r, &move_c,
					&winning_player, &node_count, &eval_count, nullptr);

				int moveIndex = ConvertToIndex(move_r, move_c, boardSide);
				myFile.write(reinterpret_cast<const char *>(&moveIndex), sizeof(int));
				myFile << std::endl;
				game.PlayMove(moveIndex);
				
				myFile.write(reinterpret_cast<const char *>(&lastMove), sizeof(int));
				myFile << std::endl;

				lastMove = moveIndex;
			}

			game.ResetBoard();
		}

		myFile.close();
	}
}