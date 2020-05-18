#include "pch.h"
#include "MonteCarloTreeSearch.h"
#include "GomokuUtils.h"

#include "MonteCarloNode.h"
#include "GomokuGame.h"

#include "torch/torch.h"

#include <stdlib.h>
#include <thread>
#include <future>
#include <vector>

namespace MonteCarlo
{
	MonteCarloTreeSearch::MonteCarloTreeSearch(int gameSpace, std::shared_ptr<GomokuPolicyAgent> pAgent, int playouts)
		: m_playouts(playouts)
		, m_gameSpace(gameSpace)
		, m_pRoot(new MonteCarloNode(nullptr, 1.0, gameSpace))
		, m_pAgent(pAgent)
	{
		if (pAgent)
		{
			m_expandChildrenFn = std::bind(&MonteCarloTreeSearch::ExpandChildrenPolicyAgent_, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			m_playoutGameFn = std::bind(&MonteCarloTreeSearch::AgentPlayout_, this, std::placeholders::_1);
		}
		else
		{
			m_expandChildrenFn = std::bind(&MonteCarloTreeSearch::DefaultExpandChildren_, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			m_playoutGameFn = std::bind(&MonteCarloTreeSearch::DefaultPlayout_, this, std::placeholders::_1);
		}
	}	

	MonteCarloTreeSearch::~MonteCarloTreeSearch()
	{
		m_pRoot->DeleteChildrenExcept(-1);
		delete m_pRoot;
	}

	/*--------------------------------------------------------------*/

	void MonteCarloTreeSearch::UpdateModel(std::string const& modelPath)
	{
		m_pAgent->LoadModel(modelPath);
	}

	void MonteCarloTreeSearch::Reset()
	{
		m_pRoot->DeleteChildrenExcept(-1);
		delete m_pRoot;
		m_pRoot = new MonteCarloNode(nullptr, 1.0, m_gameSpace);
	}

	MonteCarloNode* const MonteCarloTreeSearch::GetRoot()
	{
		return m_pRoot;
	}

	void MonteCarloTreeSearch::StepInTree(int index)
	{
		_ASSERT(index >= 0 && index < m_gameSpace);
		if (m_pRoot->GetChildrenCount() == 0) // Probably first move of the game
			return;

		if (m_pRoot->GetChildren()[index] != nullptr)
		{
			MonteCarloNode* pTemp = m_pRoot->GetChildren()[index];
			m_pRoot->DeleteChildrenExcept(index);
			delete m_pRoot;

			m_pRoot = pTemp;
		}
		else
		{
			_ASSERT(false);
			exit(-1);
		}
	}

	int MonteCarloTreeSearch::GetMove(GomokuGame const& game)
	{
		if (game.GetLastMove() == -1)
			return m_gameSpace / 2;

		for (int i = 0; i < m_playouts; i++)
		{
			GomokuGame gameCopy(game);
			m_playoutGameFn(gameCopy);
		}

		return m_pRoot->SelectVisits();
	}

	/*--------------------------------------------------------------*/

	void MonteCarloTreeSearch::AgentPlayout_(GomokuGame& game)
	{
		if (m_pRoot->GetChildrenCount() > 0)
		{
			int move = m_pRoot->Select(game.GetPlayerTurn());
			MonteCarloNode* pNode = m_pRoot->GetChildren()[move];
			_ASSERT(pNode != nullptr);
			game.PlayMove(move);
			AgentPlayoutAsync_(game, pNode, m_pAgent, 1);

/*			int rootVisists = m_pRoot->GetVisits();
			int* values = new int[4];
			m_pRoot->SelectBestFour(game.GetPlayerTurn(), values);
			std::function<void(GomokuGame&, MonteCarloNode*, std::shared_ptr<GomokuPolicyAgent> const&, int)> playoutFn =
				std::bind(&MonteCarloTreeSearch::AgentPlayoutAsync_, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
			
			std::vector<std::thread> threads;
			threads.reserve(4);

			MonteCarloNode* pNode = m_pRoot->GetChildren()[values[0]];
			_ASSERT(pNode != nullptr);
			GomokuGame gameCopy(game);
			gameCopy.PlayMove(values[0]);
			threads.push_back(std::thread(playoutFn, gameCopy, pNode, m_pAgent, 2));

			for (int i = 1; i < 4; i++)
			{
				if (values[i] == -1)
					continue;
				MonteCarloNode* pNode = m_pRoot->GetChildren()[values[i]];
				_ASSERT(pNode != nullptr);
				GomokuGame gameCopy(game);
				gameCopy.PlayMove(values[i]);
				threads.push_back(std::thread(playoutFn, gameCopy, pNode, m_pAgent, 1));
			}

			for (int i = 0; i < threads.size(); ++i)
			{
				threads[i].join();
			}
			
			rootVisists += 2;
			m_pRoot->SetVisits(rootVisists);
			delete[] values;
			*/
		}
		else
		{
			m_expandChildrenFn(m_pRoot, &game, m_pAgent);
		}
	}

	void MonteCarloTreeSearch::AgentPlayoutAsync_(
		GomokuGame& game,
		MonteCarloNode* pNode,
		std::shared_ptr<GomokuPolicyAgent> const& pAgent,
		int visitUpdate)
	{
		if (!pNode)
			return;
		while (pNode->GetChildrenCount() > 0)
		{
			int actionIndex = pNode->Select(game.GetPlayerTurn());

			{
				_ASSERT(actionIndex >= 0 && actionIndex < m_gameSpace);
				pNode = pNode->GetChildren()[actionIndex];
				_ASSERT(pNode != nullptr);
			}

			if (!game.PlayMove(actionIndex))
			{
				throw _HAS_EXCEPTIONS;
				exit(-1);
			}
		}

		std::thread expandThread;
		switch (game.GetGameWinState())
		{
		case WinnerState_enum::None:
			expandThread = std::thread(m_expandChildrenFn, pNode, &game, pAgent);
			break;
		case WinnerState_enum::P1:
			pNode->RecursiveUpdate(5.0, visitUpdate);
			return;
		case WinnerState_enum::P2:
			pNode->RecursiveUpdate(-5.0, visitUpdate);
			return;
		case WinnerState_enum::Draw:
			pNode->RecursiveUpdate(0.0, visitUpdate);
			return;
		default:
			expandThread = std::thread(m_expandChildrenFn, pNode, &game, pAgent);
			break;
		}

		double leafValue = pAgent->PredictValue(game.GetBoard(), m_gameSpace, game.GetLastMove(), game.GetPlayerTurn());

		pNode->RecursiveUpdate(leafValue, visitUpdate);
		if (expandThread.joinable())
			expandThread.join();
	}

	/*--------------------------------------------------------------*/

	void MonteCarloTreeSearch::DefaultPlayout_(GomokuGame& game)
	{
		MonteCarloNode* pNode = m_pRoot;

		while (pNode->GetChildrenCount() > 0)
		{
			int actionIndex = pNode->Select(game.GetPlayerTurn());

			{
				_ASSERT(actionIndex >= 0 && actionIndex < m_gameSpace);
				pNode = pNode->GetChildren()[actionIndex];
				_ASSERT(pNode != nullptr);
			}

			if (!game.PlayMove(actionIndex))
			{
				throw _HAS_EXCEPTIONS;
				exit(-1);
			}
		}
		
		if (game.GetGameWinState() == WinnerState_enum::None)
		{
			m_expandChildrenFn(pNode, &game, m_pAgent);
		}

		double leafValue = DefaultEvaluateRollout_(game);
		

		pNode->RecursiveUpdate(leafValue);
	}

	double MonteCarloTreeSearch::AsyncRollout_(GomokuGame* pGame, int index)
	{
		pGame->PlayMove(index);
		while (pGame->GetGameWinState() == WinnerState_enum::None)
		{
			int size = 0;
			int* pLegalMoves = pGame->GetLegalMoves(size);

			// FIXME Use better metric to rollout
			pGame->PlayMove(pLegalMoves[rand() % size]);
		}

		WinnerState_enum state = pGame->GetGameWinState();
		switch (state)
		{
		case WinnerState_enum::P1:
			return 1.0;
		case WinnerState_enum::P2:
			return -1.0;
		case WinnerState_enum::Draw:
		default:
			return 0.0;
		}
	}

	double MonteCarloTreeSearch::DefaultEvaluateRollout_(GomokuGame const& game)
	{
		WinnerState_enum state = game.GetGameWinState();
		if (state != WinnerState_enum::None)
		{
			switch (state)
			{
			case WinnerState_enum::P1:
				return 1.0;
			case WinnerState_enum::P2:
				return -1.0;
			case WinnerState_enum::Draw:
				return 0.0;
			}
		}
		int size = 0;
		int* pLegalMoves = game.GetLegalMoves(size);
		
		int* pLocalLegalCopy = new int[size];
		int localSize = size;
		memcpy(pLocalLegalCopy, pLegalMoves, size * sizeof(int));

		while (localSize > 16)
		{
			int index = rand() % size;
			if (pLocalLegalCopy[index] != -1)
			{
				pLocalLegalCopy[index] = -1;
				localSize--;
			}
		}

		std::vector<std::future<double>> pPromises;
		pPromises.resize(localSize);
		GomokuGame** ppGames = new GomokuGame*[localSize];
		int promiseIndex = 0;
		for (int i = 0; i < size; i++)
		{
			if (pLocalLegalCopy[i] != -1)
			{
				GomokuGame* pLocalGame = new GomokuGame(game);
				ppGames[promiseIndex] = pLocalGame;
				pPromises[promiseIndex] = std::async(&MonteCarloTreeSearch::AsyncRollout_, this, pLocalGame, pLegalMoves[i]);
				promiseIndex++;
			}
		}
		
		double EvalTotal = 0.0;
		for (int i = 0; i < localSize; i++)
		{
			EvalTotal += pPromises[i].get();
			delete ppGames[i];
		}

		delete ppGames;

		return EvalTotal / localSize;
	}

	/*--------------------------------------------------------------*/

	void MonteCarloTreeSearch::ExpandChildrenPolicyAgent_(
		MonteCarloNode* pNode,
		GomokuGame* pGame,
		std::shared_ptr<GomokuPolicyAgent> const& pAgent)
	{
		int size = 0;
		int* pLegalMoves = pGame->GetLegalMoves(size);

		torch::Tensor& agentPolicy = pAgent->PredictMove(pGame->GetBoard(), m_gameSpace, pGame->GetLastMove(), pGame->GetPlayerTurn());

		pNode->ExpandChildren(pLegalMoves, agentPolicy, size);
	}

	void MonteCarloTreeSearch::DefaultExpandChildren_(
		MonteCarloNode* pNode,
		GomokuGame* pGame,
		std::shared_ptr<GomokuPolicyAgent> const& /*pAgent*/)
	{
		int size = 0;
		int* pLegalMoves = pGame->GetLegalMoves(size);

		pNode->DefaultExpandChildren(pLegalMoves, size);
	}
}