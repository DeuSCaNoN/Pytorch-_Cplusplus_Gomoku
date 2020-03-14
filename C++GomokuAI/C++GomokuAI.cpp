// C++GomokuAI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "GomokuGame.h"
#include "MonteCarloTreeSearch.h"
#include "GomokuPolicyAgent.h"
#include "GomokuUtils.h"

#include <iostream>

void DrawMatrix(char** matrix, int sideLength)
{
	std::cout << "\n";
	for (int i = 0; i < sideLength; i++)
	{
		for (int j = 0; j < sideLength; j++)
		{
			char output = 0;
			switch (matrix[i][j])
			{
			case 0:
				break;
			case 1:
				output = '1';
				break;
			case 2:
				output = '2';
			}
			std::cout << output << " | ";
		}
		std::cout << "\n";
	}
}

int main()
{
	short boardSize = 15;
	short win = 5;

	GomokuGame game = GomokuGame(boardSize, win);
	game.PlayMove(7,7);
	game.PlayMove(6, 7);
	game.PlayMove(7, 6);
	game.PlayMove(6, 6);
	GomokuPolicyAgent agent;
	agent.PredictMove(game.GetBoard(), 225, ConvertToIndex(6,6,15), true);
	
	MonteCarlo::MonteCarloTreeSearch treeSearch(boardSize*boardSize, 20);
	int index = treeSearch.GetMove(game);
	game.PlayMove(index);
	DrawMatrix(game.GetMatrix(), boardSize);

	return game.IsMoveWinning(2, 2);

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
