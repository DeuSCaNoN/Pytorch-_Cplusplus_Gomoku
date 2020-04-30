// C++GomokuAI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "GomokuGame.h"
#include "MonteCarloTreeSearch.h"
#include "GomokuPolicyAgent.h"
#include "GomokuUtils.h"

#include <iostream>

int main()
{
	GomokuUtils::TrainSelfPlay();

/*	short boardSize = 15;
	short win = 5;

	std::shared_ptr<GomokuPolicyAgent> pAgent = std::make_shared<GomokuPolicyAgent>();
//	MonteCarlo::MonteCarloTreeSearch treeSearch((int)(boardSize*boardSize), pAgent, 200);

	GomokuGame game = GomokuGame(boardSize, win);
	game.PlayMove(112);
	game.PlayMove(0);
	while (game.GetGameWinState() == WinnerState_enum::None)
	{
		int index = treeSearch.GetMove(game);
		system("CLS");
		game.PlayMove(index);
		treeSearch.StepInTree(index);
		GomokuUtils::DrawMatrix(game.GetBoard(), boardSize);

		index = treeSearch.GetMove(game);
		game.PlayMove(index);
		GomokuUtils::DrawMatrix(game.GetBoard(), boardSize);
		treeSearch.StepInTree(index);
	}

	std::cout << pAgent->PredictValue(game.GetBoard(), 225, 0, true);
*/
	return 0;

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
