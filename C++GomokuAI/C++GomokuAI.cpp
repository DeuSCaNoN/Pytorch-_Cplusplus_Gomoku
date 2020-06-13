// C++GomokuAI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "GomokuGame.h"
#include "MonteCarloTreeSearch.h"
#include "GomokuPolicyAgent.h"
#include "GomokuUtils.h"
#include "AgentPlayer.h"

void Evaluate()
{
	std::shared_ptr<GomokuPolicyAgent> pAgent = std::make_shared<GomokuPolicyAgent>();
	auto pAgentPlayer = std::make_shared<Player::AgentPlayer>(pAgent, 1000);

	std::shared_ptr<GomokuPolicyAgent> pOldAgent = std::make_shared<GomokuPolicyAgent>("GomokuModel_Old.pt");
	auto pOldAgentPlayer = std::make_shared<Player::AgentPlayer>(pOldAgent, 1000);

	GomokuUtils::Evaluate(pAgentPlayer, pOldAgentPlayer);
}

void HumanSelection()
{
	std::cout << "1: Human P1" << std::endl
		<< "2: Human P2 " << std::endl
		<< "Enter number: ";
	int input;
	std::cin >> input;
	while (input < 1 || input > 2)
	{
		std::cout << "Error invalid number try again: ";
		std::cin >> input;
	}

	GomokuUtils::HumanPlay(input == 1);
}

void StartGomokuTraining()
{
	while (true)
	{
		std::cout << "1: Self train" << std::endl
			<< "2: Blupig train" << std::endl
			<< "3: Human play" << std::endl
			<< "4: Mixed Self blupig" << std::endl
			<< "5: Evaluate against old agent" << std::endl
			<< "6: Exit" << std::endl
			<< "Enter number: ";
		int input;
		std::cin >> input;
		while (input < 1 || input > 6)
		{
			std::cout << "Error invalid number try again: ";
			std::cin >> input;
		}

		system("CLS");

		if (input == 6)
			break;

		switch (input)
		{
		case 1:
			GomokuUtils::TrainSelfPlay();
			break;
		case 2:
			GomokuUtils::TrainBluPig();
			break;
		case 3:
			HumanSelection();
			break;
		case 4:
			GomokuUtils::MixedTraining();
			break;
		case 5:
			Evaluate();
			break;
		default:
			break;
		}
	}
}

int main()
{
	StartGomokuTraining();

/*	short boardSize = 15;
	short win = 5;

	std::shared_ptr<GomokuPolicyAgent> pAgent = std::make_shared<GomokuPolicyAgent>();
	MonteCarlo::MonteCarloTreeSearch treeSearch((int)(boardSize*boardSize), pAgent, 200);

	GomokuGame game = GomokuGame(boardSize, win);
	game.PlayMove(112);
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

	std::vector<TrainingExample> trainingExamples(2);
	memcpy(trainingExamples[0].board, game.GetBoard(), 225);
	memset(trainingExamples[0].pMoveEstimate, 0, BOARD_LENGTH * sizeof(float));
	trainingExamples[0].pMoveEstimate[96]= 1.0f;
	trainingExamples[0].boardValue = 0.0f;
	trainingExamples[0].lastMove = 112;

	game.PlayMove(0);
	memcpy(trainingExamples[1].board, game.GetBoard(), 225);
	memset(trainingExamples[1].pMoveEstimate, 0, BOARD_LENGTH*sizeof(float));
	trainingExamples[1].pMoveEstimate[96] = 1.0f;
	trainingExamples[1].boardValue = 1.0f;
	trainingExamples[1].lastMove = 0;
	
	pAgent->Train(trainingExamples, 0.2, 20000000);

	std::cout << pAgent->PredictValue(trainingExamples[0].board, 225, 112, false) << std::endl;
	std::cout << pAgent->PredictValue(trainingExamples[1].board, 225, 0, true) << std::endl;
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
