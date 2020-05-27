#include "pch.h"
#include "GomokuUtils.h"
#include "GomokuGame.h"
#include "GomokuPolicyAgent.h"
#include <BluPigPlayer.h>
#include <AgentPlayer.h>

#include <api/renju_api.h>
#include <ai/eval.h>

#define MAX_EXAMPLE_SIZE 5000

namespace GomokuUtils
{
	std::string trainingFilename = "TrainingLog.log";
	std::ofstream k_trainingLog;

	void WriteExampleSetToFile_(std::vector<TrainingExample> exampleSet, short fileNameNum);

	std::vector<TrainingExample> GetExampleSetFromFile_(short fileNameNum);

	bool GetAllExampleSets_(std::vector<TrainingExample>& outputVector);

	std::vector<TrainingExample> GenerateExamplesFromPlay_(PlayGeneratorCfg const& cfg, std::shared_ptr<GomokuGame> const& pGame);

	void ConvertToRowCol(
		int index,
		int sideLength,
		short& row, /*out*/
		short& col /*out*/)
	{
		row = index / sideLength;
		col = index % sideLength;
	}

	void HumanPlay()
	{
		std::shared_ptr<GomokuPolicyAgent> pAgent = std::make_shared<GomokuPolicyAgent>();
		auto pAgentPlayer = std::make_shared<Player::AgentPlayer>(pAgent, 1500);

		auto pGame = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);

		while (pGame->GetGameWinState() == WinnerState_enum::None)
		{
			float throwAway[BOARD_LENGTH];
			int index = pAgentPlayer->MakeMove(pGame, pGame->GetPlayerTurn(), throwAway);
			pGame->PlayMove(index);
			pAgentPlayer->MoveMadeInGame(index);
			system("CLS");
			GomokuUtils::DrawMatrix(pGame->GetBoard(), BOARD_SIDE);

			if (pGame->GetGameWinState() != WinnerState_enum::None)
				break;

			int row, col;
			std::cout << "Row: " << std::endl;
			std::cin >> row;
			std::cout << "Col: " << std::endl;
			std::cin >> col;
			index = ConvertToIndex(row, col, BOARD_SIDE);
			pGame->PlayMove(index);
			system("CLS");
			GomokuUtils::DrawMatrix(pGame->GetBoard(), BOARD_SIDE);
			pAgentPlayer->MoveMadeInGame(index);
		}

		if (pGame->GetGameWinState() == WinnerState_enum::P1)
			std::cout << "P1 won" << std::endl;
		if (pGame->GetGameWinState() == WinnerState_enum::P2)
			std::cout << "P2 won" << std::endl;
		if (pGame->GetGameWinState() == WinnerState_enum::Draw)
			std::cout << "DRAW" << std::endl;
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
		agent.Train(exampleSet, 0.02);

		agent.SaveModel();
	}

	/*------------------------------------------------------------------------------------------------*/

	void TrainBluPig()
	{
		k_trainingLog.open(trainingFilename, std::ofstream::binary);
		std::shared_ptr<GomokuPolicyAgent> pAgent = std::make_shared<GomokuPolicyAgent>();
		auto pBluPigPlayer = std::make_shared<Player::BluPigPlayer>();
		std::vector<TrainingExample> exampleSet;
		auto pGame1 = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
		auto pGame2 = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);

		bool bRun = true;
		std::thread signalThread([&]() {
			std::cin.get();
			bRun = !bRun;
		});
		while (bRun)
		{
			for (unsigned int i = 0; i < 100; i++)
			{
				exampleSet.clear();
				pGame1->ResetBoard();
				pGame2->ResetBoard();
				std::future<std::vector<TrainingExample>> promise1;
				std::future<std::vector<TrainingExample>> promise2;

				auto pAgentPlayer = std::make_shared<Player::AgentPlayer>(pAgent, 150);
				PlayGeneratorCfg cfg({ 0, 1, false, false, true, true, pBluPigPlayer, pAgentPlayer, true, BOARD_LENGTH / 2 });

				promise1 = std::async(&GenerateExamplesFromPlay_, cfg, pGame1);

				auto pAgentPlayer2 = std::make_shared<Player::AgentPlayer>(pAgent, 150);
				PlayGeneratorCfg cfg2({ 0, 1, false, false, true, true, pAgentPlayer2, pBluPigPlayer, true, BOARD_LENGTH / 2 });

				promise2 = std::async(&GenerateExamplesFromPlay_, cfg2, pGame2);

				std::vector<TrainingExample> subExampleSet = promise1.get();
				exampleSet.insert(exampleSet.end(), subExampleSet.begin(), subExampleSet.end());

				subExampleSet = promise2.get();
				exampleSet.insert(exampleSet.end(), subExampleSet.begin(), subExampleSet.end());

				pAgent->Train(exampleSet, 0.2);
				pAgent->SaveModel();
			}

			auto pAgentPlayer = std::make_shared<Player::AgentPlayer>(pAgent, 800);
			Evaluate(pAgentPlayer, pBluPigPlayer);
			std::ifstream src(pAgent->GetModelPath(), std::ios::binary);
			std::ofstream dst("GomokuModel_Old.pt", std::ios::binary);

			dst << src.rdbuf();
		}

		signalThread.join();
	}

	void TrainSelfPlay()
	{
		std::shared_ptr<GomokuPolicyAgent> pAgent = std::make_shared<GomokuPolicyAgent>();
		std::vector<TrainingExample> exampleSet;

		//unsigned supportedGames = std::thread::hardware_concurrency() / 4;

		auto pGame = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
		auto pAgentPlayer = std::make_shared<Player::AgentPlayer>(pAgent, 600);
		unsigned int count = 1;

		bool bRun = true;
		std::thread signalThread([&]() {
			std::cin.get();
			bRun = !bRun;
		});
		while (bRun)
		{
			pGame->ResetBoard();
			pAgentPlayer->ClearTree();

			PlayGeneratorCfg cfg({ 0, 1, false, false, true, true, pAgentPlayer, pAgentPlayer, true, BOARD_LENGTH / 2 });
			exampleSet = std::move(GenerateExamplesFromPlay_(cfg, pGame));

			pAgent->Train(exampleSet, 0.2);
			pAgent->SaveModel();
			// pAgentPlayer->UpdateModel(pAgent->GetModelPath()); Don't need this yet
			
			if (count == 100)
			{
				std::ifstream src(pAgent->GetModelPath(), std::ios::binary);
				std::ofstream dst("GomokuModel_Old.pt", std::ios::binary);

				dst << src.rdbuf();
				count = 1;
			}
			exampleSet.clear();
		}

		signalThread.join();
	}

	void Evaluate(std::shared_ptr<Player::IPlayer> pAgent1, std::shared_ptr<Player::IPlayer> pAgent2)
	{
		std::cout << "Evaluating" << std::endl;
		k_trainingLog << "Evaluating" << std::endl;
		std::future<std::vector<TrainingExample>> game1Promise;

		auto pGame = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
		PlayGeneratorCfg cfg({ 0, 1, false, false, true, true, pAgent1, pAgent2, true, -1 });
		std::vector<TrainingExample> subExampleSet = GenerateExamplesFromPlay_(cfg, pGame);

		switch (pGame->GetGameWinState())
		{
		case WinnerState_enum::P1:
			k_trainingLog << "Player 1 " + pAgent1->PrintWinningStatement() << std::endl;
			break;
		case WinnerState_enum::P2:
			k_trainingLog << "Player 2 " + pAgent2->PrintWinningStatement() << std::endl;
			break;
		default:
			break;
		}

		pAgent1->ClearTree();
		pAgent2->ClearTree();

		auto pGame2 = std::make_shared<GomokuGame>(BOARD_SIDE, BOARD_WIN);
		PlayGeneratorCfg cfg2({ 0, 1, false, false, true, false, pAgent2, pAgent1, true, -1 });
		subExampleSet = GenerateExamplesFromPlay_(cfg2, pGame2);

		switch (pGame->GetGameWinState())
		{
		case WinnerState_enum::P1:
			k_trainingLog << "Player 1 " + pAgent2->PrintWinningStatement() << std::endl;
			break;
		case WinnerState_enum::P2:
			k_trainingLog << "Player 2 " + pAgent1->PrintWinningStatement() << std::endl;
			break;
		default:
			break;
		}

		std::cout << "Evaluating Done" << std::endl;
		k_trainingLog << "Evaluating Done" << std::endl;
	}

	/*------------------------------------------------------------------------------------------------*/

	std::vector<TrainingExample> GenerateExamplesFromPlay_(PlayGeneratorCfg const& cfg, std::shared_ptr<GomokuGame> const& pGame)
	{
		int gameSpace = pGame->GetSideLength() * pGame->GetSideLength();
		std::vector<TrainingExample> exampleSet(MAX_EXAMPLE_SIZE);
		int exampleSize = 0;
		srand((unsigned int)time(NULL) + cfg.seed);

		for (unsigned int i = 0; i < cfg.gameCount; i++)
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

				int move;
				if (bTurn)
					move = cfg.pPlayer1->MakeMove(pGame, bTurn, exampleSet[exampleSize].pMoveEstimate);
				else
					move = cfg.pPlayer2->MakeMove(pGame, bTurn, exampleSet[exampleSize].pMoveEstimate);

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
					lastMove = move;
				}

				if (bSave && exampleSize < MAX_EXAMPLE_SIZE - 1)
					++exampleSize;

				bFirst = false;
			}

			float maxboardValue = 0.0f;
			switch (pGame->GetGameWinState())
			{
			case WinnerState_enum::P1:
				maxboardValue = 1.0 * 9;
				k_trainingLog << "Player 1 " + cfg.pPlayer1->PrintWinningStatement() << std::endl;
				break;
			case WinnerState_enum::P2:
				maxboardValue = -1.0 * 10;
				k_trainingLog << "Player 2 " + cfg.pPlayer2->PrintWinningStatement() << std::endl;
				break;
			default:
				break;
			}

			float boardValue = maxboardValue / pGame->GetMovesPlayed();
			float boardValueIncrease = (maxboardValue / abs(maxboardValue) - boardValue) / (exampleSize - currentGameStart - 1);
			for (int j = currentGameStart; j < exampleSize; j++)
			{
				exampleSet[j].boardValue = boardValue;
				boardValue += boardValueIncrease;
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