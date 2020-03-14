#include "pch.h"
#include "GomokuUtils.h"

#include "GomokuGame.h"

#include <api/renju_api.h>
#include <ai/eval.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <vector>
#include <thread>

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
					&winning_player, &node_count, &eval_count, nullptr, nullptr);

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

	void GenerateValueSetAsync_(short boardSide, int fileNameNum)
	{
		GomokuGame game = GomokuGame(boardSide, 5);

		std::ofstream myFile;
		std::string fileName = "valueDataset" + std::to_string(fileNameNum) + ".txt";
		myFile.open(fileName, std::ofstream::binary);
		srand((unsigned int)time(NULL) + fileNameNum); 

		for (int i = 0; i < 200; i++)
		{
			int player = 0;
			int lastMove = 0;
			short initialMoves = rand() % 10 + 1;
			for (int j = 0; j < initialMoves; j++)
			{
				int size = 0;
				int* const pLegalMoves = game.GetLegalMoves(size);
				int initialMove = pLegalMoves[rand() % (size)];
				game.PlayMove(initialMove);
				lastMove = initialMove;
				player = (player + 1) % 2;
			}

			while (game.GetGameWinState() == WinnerState_enum::None)
			{
				myFile.write(game.GetBoard(), boardSide * boardSide);
				myFile << std::endl;

				char* moveValues = new char[boardSide*boardSide];
				memset(moveValues, 0, boardSide*boardSide);
				int move_r, move_c, winning_player, actual_depth;
				unsigned int node_count, eval_count;

				bool success = RenjuAPI::generateMove(game.GetBoard(), player + 1, -1, 1500, 1, &actual_depth, &move_r, &move_c,
					&winning_player, &node_count, &eval_count, nullptr, moveValues);

				if (!success || move_r < 0 || move_c < 0)
				{
					delete[] moveValues;
					myFile.close();
					std::cout << "RENJU API ERROR!";
					exit(10);
				}

				unsigned char max_score = moveValues[ConvertToIndex(move_r, move_c, boardSide)];
				if (max_score == 0)
				{
					moveValues[ConvertToIndex(move_r, move_c, boardSide)] = 1;
					max_score = 1;
				}
				myFile.write(reinterpret_cast<const char *>(&max_score), sizeof(unsigned char));
				myFile << std::endl;

				myFile.write(moveValues, boardSide * boardSide);
				myFile << std::endl;

				double stateVal = RenjuAIEval::evalState(game.GetBoard(), player + 1) >> 1;
				if (stateVal < 0)
					stateVal = 0.0;

				stateVal = log(stateVal) / log(10000);
				if (stateVal > 1)
					stateVal = 1.0;
				else if (stateVal < 0)
					stateVal = 0.0;

				myFile << std::to_string(stateVal) << std::endl;

				if (move_r > boardSide || move_c > boardSide)
				{
					std::cout << "ERROR move too big";
					delete[] moveValues;
					myFile.close();
					exit(10);
				}
				int moveIndex = ConvertToIndex(move_r, move_c, boardSide);

				game.PlayMove(moveIndex);
				player = (player + 1) % 2;

				myFile.write(reinterpret_cast<const char *>(&lastMove), sizeof(int));
				myFile << std::endl;

				lastMove = moveIndex;

				delete[] moveValues;
			}
			std::cout << i << std::endl;
			game.ResetBoard();
		}

		myFile.close();
	}

	void GenerateValueSet(short boardSide)
	{
		std::vector<std::thread> threads;
		threads.reserve(15);
		for (int i = 1; i < 16; i++)
		{
			auto thread = std::thread(GenerateValueSetAsync_, boardSide, i);
			threads.push_back(std::move(thread));
		}

		GenerateValueSetAsync_(boardSide, 0);

		for (int i = 0; i < threads.size(); i++)
		{
			threads[i].join();
		}
	}
}