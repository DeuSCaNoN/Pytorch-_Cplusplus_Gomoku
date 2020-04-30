#include "pch.h"
#include "GomokuUtils.h"
#include "GomokuGame.h"
#include "GomokuPolicyAgent.h"
#include <BluPigPlayer.h>
#include <AgentPlayer.h>

#include <api/renju_api.h>
#include <ai/eval.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <vector>
#include <future>
#include <thread>

#define MAX_EXAMPLE_SIZE 5000

namespace GomokuUtils
{
	void WriteExampleSetToFile_(std::vector<TrainingExample> exampleSet, short fileNameNum);

	std::vector<TrainingExample> GetExampleSetFromFile_(short fileNameNum);

	bool GetAllExampleSets_(std::vector<TrainingExample>& outputVector);

	std::vector<TrainingExample> GenerateExamplesFromPlay_(PlayGeneratorCfg const& cfg, std::shared_ptr<GomokuGame> pGame);

	void ConvertToRowCol(
		int index,
		int sideLength,
		short& row, /*out*/
		short& col /*out*/)
	{
		row = index / sideLength;
		col = index % sideLength;
	}

	void DrawMatrix(char* matrix, int sideLength)
	{
		std::cout << "\n" << "   ";
		for (int i = 0; i < sideLength; i++)
		{
			std::cout << i << "  ";
			if (i < 10)
				std::cout << " ";
		}
		std::cout << std::endl;
		for (int i = 0; i < sideLength; i++)
		{
			std::cout << i << " ";
			if (i < 10)
				std::cout << " ";
			for (int j = 0; j < sideLength; j++)
			{
				char output = 0;
				switch (matrix[i*sideLength + j])
				{
				case 0:
					break;
				case 1:
					output = 'X';
					break;
				case 2:
					output = 'O';
				}
				std::cout << output << " | ";
			}
			std::cout << "\n";
		}
	}

	void TeachFromValueSet(bool bGenerate)
	{
		std::vector<TrainingExample> exampleSet;
		bool bExamplesFound = false;
		if (!bGenerate)
		{
			bExamplesFound = GetAllExampleSets_(exampleSet);
		}
		
		if (!bExamplesFound)
		{
			unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
			std::vector<std::future<std::vector<TrainingExample>>> pPromises;
			pPromises.reserve(concurentThreadsSupported - 1);
			auto pBluPigPlayer = std::make_shared<Player::BluPigPlayer>();

			for (unsigned short i = 1; i < concurentThreadsSupported; i++)
			{
				auto pGame = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
				PlayGeneratorCfg cfg({i, 500, true, true, true, true, pBluPigPlayer, pBluPigPlayer, false, -1 });
				pPromises.push_back(std::async(&GenerateExamplesFromPlay_, cfg, pGame));
			}

			auto pGame = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
			PlayGeneratorCfg cfg({ 0,500, true, true, true, true, pBluPigPlayer, pBluPigPlayer, false, -1 });
			exampleSet = GenerateExamplesFromPlay_(cfg, pGame);
			WriteExampleSetToFile_(exampleSet, 15);

			std::vector<std::thread> threads;
			threads.reserve(pPromises.size());
			for (int i = 0; i < pPromises.size(); i++)
			{
				std::vector<TrainingExample> subExampleSet = pPromises[i].get();
				auto thread = std::thread(WriteExampleSetToFile_, subExampleSet, i);
				threads.push_back(std::move(thread));

				exampleSet.insert(exampleSet.end(), subExampleSet.begin(), subExampleSet.end());
			}

			for (int i = 0; i < threads.size(); i++)
			{
				threads[i].join();
			}
		}

		GomokuPolicyAgent agent = GomokuPolicyAgent();
		agent.Train(exampleSet, 0.02, 15);

		agent.SaveModel();
	}

	/*------------------------------------------------------------------------------------------------*/

	void TrainBluPig()
	{
		std::shared_ptr<GomokuPolicyAgent> pAgent = std::make_shared<GomokuPolicyAgent>();
		auto pBluPigPlayer = std::make_shared<Player::BluPigPlayer>();
		std::vector<TrainingExample> exampleSet;
		
		while (true)
		{
			exampleSet.clear();
			std::future<std::vector<TrainingExample>> promise1;
			std::future<std::vector<TrainingExample>> promise2;

			auto pAgentPlayer = std::make_shared<Player::AgentPlayer>(pAgent, 200);
			PlayGeneratorCfg cfg({ 0, 1, true, false, true, true, pBluPigPlayer, pAgentPlayer, true, -2 });
			auto pGame1 = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
			promise1 = std::async(&GenerateExamplesFromPlay_, cfg, pGame1);

			auto pAgentPlayer2 = std::make_shared<Player::AgentPlayer>(pAgent, 200);
			PlayGeneratorCfg cfg2({ 0, 1, true, false, true, true, pAgentPlayer2, pBluPigPlayer, true, -2 });
			auto pGame2 = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
			promise2 = std::async(&GenerateExamplesFromPlay_, cfg2, pGame2);

			auto pGame3 = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
			PlayGeneratorCfg cfg3({ 0, 1, true, false, true, true, pBluPigPlayer, pBluPigPlayer, true, -1 });
			exampleSet = GenerateExamplesFromPlay_(cfg3, pGame3);

			std::vector<TrainingExample> subExampleSet = promise1.get();
			exampleSet.insert(exampleSet.end(), subExampleSet.begin(), subExampleSet.end());

			subExampleSet = promise2.get();
			exampleSet.insert(exampleSet.end(), subExampleSet.begin(), subExampleSet.end());

			pAgent->Train(exampleSet, 0.002, 20);
			pAgent->SaveModel();
		}
	}

	void TrainSelfPlay()
	{
		std::shared_ptr<GomokuPolicyAgent> pAgent = std::make_shared<GomokuPolicyAgent>();
		std::vector<TrainingExample> exampleSet;

		unsigned supportedGames = std::thread::hardware_concurrency() / 4;
		int gameSpace = BOARD_SIDE * BOARD_SIDE;
		std::vector<int> startMove = { gameSpace / 2, gameSpace / 4, gameSpace*3/4, gameSpace / 4 + 4 };
		short trainingCount = 0;
		while (true)
		{
			exampleSet.clear();
			std::vector<std::future<std::vector<TrainingExample>>> pPromises;
			pPromises.reserve(supportedGames);

			for (unsigned short i = 0; i < supportedGames; i++)
			{
				auto pGame = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
				auto pAgentPlayer = std::make_shared<Player::AgentPlayer>(pAgent, 200);
				PlayGeneratorCfg cfg({ i, 1, true, false, true, true, pAgentPlayer, pAgentPlayer, true, startMove[i] });
				pPromises.push_back(std::async(&GenerateExamplesFromPlay_, cfg, pGame));
			}

			for (int i = 0; i < pPromises.size(); i++)
			{
				std::vector<TrainingExample> subExampleSet = pPromises[i].get();
				exampleSet.insert(exampleSet.end(), subExampleSet.begin(), subExampleSet.end());
			}

			pAgent->Train(exampleSet, 0.002, 7);
			pAgent->SaveModel();

			trainingCount++;
			if (trainingCount == 50)
			{
				Evaluate();
				trainingCount = 0;
				std::ifstream src(pAgent->GetModelPath(), std::ios::binary);
				std::ofstream dst("GomokuModel_Old.pt", std::ios::binary);

				dst << src.rdbuf();
			}
		}
	}

	short Evaluate()
	{
		std::cout << "Evaluating" << std::endl;
		std::shared_ptr<GomokuPolicyAgent> pAgent = std::make_shared<GomokuPolicyAgent>();
		std::shared_ptr<GomokuPolicyAgent> pAgentOld = std::make_shared<GomokuPolicyAgent>("GomokuModel_Old.pt");

		std::future<std::vector<TrainingExample>> game1Promise;
		auto pAgentPlayer = std::make_shared<Player::AgentPlayer>(pAgent, 200);
		auto pAgentPlayerOld = std::make_shared<Player::AgentPlayer>(pAgentOld, 200);

		auto pGame = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
		PlayGeneratorCfg cfg({ 0, 1, false, false, true, true, pAgentPlayer, pAgentPlayerOld, true, -1 });
		std::vector<TrainingExample> subExampleSet = GenerateExamplesFromPlay_(cfg, pGame);

		short winAmount = 0;
		if (subExampleSet.size() > 1)
		{
			bool bAgentWon = subExampleSet[0].boardValue > 0;
			std::cout << "P1: Agent vs P2: Old Agent " << bAgentWon << std::endl;

			if (bAgentWon)
				winAmount++;
		}

		pAgentPlayer->ClearTree();
		pAgentPlayerOld->ClearTree();

		auto pGame2 = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
		PlayGeneratorCfg cfg2({ 0, 1, false, false, true, false, pAgentPlayerOld, pAgentPlayer, true, -1 });
		subExampleSet = GenerateExamplesFromPlay_(cfg2, pGame2);

		if (subExampleSet.size() > 1)
		{
			bool bAgentWon = subExampleSet[0].boardValue < 0;
			std::cout << "P1: Old Agent vs P2: Agent " << bAgentWon << std::endl;

			if (bAgentWon)
				winAmount++;
		}

		std::cout << "Evaluating Done" << std::endl;
		return winAmount;
	}

	/*------------------------------------------------------------------------------------------------*/

	std::vector<TrainingExample> GenerateExamplesFromPlay_(PlayGeneratorCfg const& cfg, std::shared_ptr<GomokuGame> pGame)
	{
		int gameSpace = pGame->GetSideLength() * pGame->GetSideLength();
		std::vector<TrainingExample> exampleSet(MAX_EXAMPLE_SIZE);
		int exampleSize = 0;
		srand((unsigned int)time(NULL) + cfg.seed);

		for (int i = 0; i < cfg.gameCount; i++)
		{
			int lastMove = -1;
			int currentGameStart = exampleSize;
			if (cfg.startMove >= 0)
			{
				pGame->PlayMove(cfg.startMove);
				lastMove = cfg.startMove;
			}
			else
			{
				short initialMoves = cfg.bRandStart ? rand() % 10 + 1 : 0;
				for (int j = 0; j < initialMoves; j++)
				{
					int size = 0;
					int* const pLegalMoves = pGame->GetLegalMoves(size);
					int initialMove = pLegalMoves[rand() % (size)];
					pGame->PlayMove(initialMove);
					lastMove = initialMove;
				}
			}
			
			bool bFirst = true;

			while (pGame->GetGameWinState() == WinnerState_enum::None)
			{
				bool bSave = true;
				bool bTurn = pGame->GetPlayerTurn();
				if (bFirst && cfg.bRandStart && cfg.startMove == -2)
					bSave = false; 
				else if (bTurn)
					bSave = bSave & cfg.bSavePlayer1;
				else
					bSave = bSave & cfg.bSavePlayer2;

				memcpy(exampleSet[exampleSize].board, pGame->GetBoard(), gameSpace);

				int move, moveToSave;
				if (bTurn)
					move = cfg.pPlayer1->MakeMove(pGame, bTurn, moveToSave);
				else
					move = cfg.pPlayer2->MakeMove(pGame, bTurn, moveToSave);

				exampleSet[exampleSize].moveMade = moveToSave;
				exampleSet[exampleSize].boardValue = 0.0;

				if (move < 0 || move > pGame->GetSideLength() * pGame->GetSideLength())
				{
					std::cout << "ERROR move out of bounds";
					exit(10);
				}

				if (cfg.bRandMoves && (rand() % 4) == 0)
				{
					int size = 0;
					int* const pLegalMoves = pGame->GetLegalMoves(size);
					int randomMove = pLegalMoves[rand() % (size)];
					pGame->PlayMove(randomMove);
					cfg.pPlayer1->MoveMadeInGame(randomMove);
					cfg.pPlayer2->MoveMadeInGame(randomMove);

					exampleSet[exampleSize].lastMove = lastMove;
					lastMove = randomMove;
				}
				else
				{
					pGame->PlayMove(move);
					cfg.pPlayer1->MoveMadeInGame(move);
					cfg.pPlayer2->MoveMadeInGame(move);

					exampleSet[exampleSize].lastMove = lastMove;
					lastMove = exampleSet[exampleSize].moveMade;
				}

				if (bSave && exampleSize < MAX_EXAMPLE_SIZE - 1)
					++exampleSize;

				bFirst = false;
			}

			float boardValue = 0.0;
			switch (pGame->GetGameWinState())
			{
			case WinnerState_enum::P1:
				boardValue = 1.0 * 9;
				break;
			case WinnerState_enum::P2:
				boardValue = -1.0 * 10;
				break;
			default:
				break;
			}

			boardValue = boardValue / pGame->GetMovesPlayed();
			for (int j = currentGameStart; j < exampleSize; j++)
			{
				exampleSet[j].boardValue = boardValue;
			}

			if (exampleSize == MAX_EXAMPLE_SIZE - 1)
				exampleSize++;

			if (exampleSize == MAX_EXAMPLE_SIZE)
				break;

			if (cfg.bPrint)
				DrawMatrix(pGame->GetBoard(), pGame->GetSideLength());

			pGame->ResetBoard();
		}

		exampleSet.resize(exampleSize);
		return exampleSet;
	}

	void WriteExampleSetToFile_(std::vector<TrainingExample> exampleSet, short fileNameNum)
	{
		std::string fileName = "valueDataset" + std::to_string(fileNameNum) + ".bin";
		std::ofstream exampleFile;
		exampleFile.open(fileName, std::ofstream::binary);

		for (int i = 0; i < exampleSet.size(); i++)
		{
			TrainingExample const& test = exampleSet[i];
			exampleFile.write((char*)&test, sizeof(TrainingExample));
		}
	}

	std::vector<TrainingExample> GetExampleSetFromFile_(short fileNameNum)
	{
		std::string fileName = "valueDataset" + std::to_string(fileNameNum) + ".bin";
		std::vector<TrainingExample> returnVect;
		std::ifstream exampleFile(fileName, std::ifstream::binary);

		returnVect.reserve(MAX_EXAMPLE_SIZE);
		for (int i = 0; i < MAX_EXAMPLE_SIZE; i++)
		{
			TrainingExample temp;
			exampleFile.read((char*)&temp, sizeof(TrainingExample));
			if (exampleFile.eof() || !exampleFile.good())
				break;

			returnVect.push_back(temp);
		}

		return returnVect;
	}

	bool GetAllExampleSets_(std::vector<TrainingExample>& outputVector)
	{
		short i = 0;
		std::string fileName = "valueDataset" + std::to_string(i) + ".bin";
		std::ifstream exampleFile(fileName, std::ifstream::binary);
		std::vector<std::future<std::vector<TrainingExample>>> pPromises;
		
		while(exampleFile.good())
		{
			pPromises.push_back(std::async(&GetExampleSetFromFile_, i));
			++i;
			exampleFile.close();
			fileName = "valueDataset" + std::to_string(i) + ".bin";
			exampleFile = std::ifstream(fileName, std::ifstream::binary);
		}

		if (i == 0)
			return false;

		for (int i = 0; i < pPromises.size(); i++)
		{
			std::vector<TrainingExample> subExampleSet = pPromises[i].get();
			outputVector.insert(outputVector.end(), subExampleSet.begin(), subExampleSet.end());
		}

		return true;
	}
}