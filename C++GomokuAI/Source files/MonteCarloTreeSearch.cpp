#include "pch.h"
#include "MonteCarloTreeSearch.h"
#include "GomokuUtils.h"

#include "MonteCarloNode.h"
#include "GomokuGame.h"


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
			m_playoutGameFn = std::bind(&MonteCarloTreeSearch::AgentPlayout_, this, std::placeholders::_1);
		}
		else
		{
			m_playoutGameFn = std::bind(&MonteCarloTreeSearch::DefaultPlayout_, this, std::placeholders::_1);
		}
	}	

	MonteCarloTreeSearch::~MonteCarloTreeSearch()
	{
		if (m_deleteThread.joinable())
			m_deleteThread.join();
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
		if (m_pRoot->GetChildrenCount() == 0)
			return;

		if (m_deleteThread.joinable())
			m_deleteThread.join();

		m_deleteThread = std::thread(&MonteCarloTreeSearch::DeleteNodes_, this, -1, m_pRoot);
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
			if (m_deleteThread.joinable())
				m_deleteThread.join();

			m_deleteThread = std::thread(&MonteCarloTreeSearch::DeleteNodes_, this, index, m_pRoot);

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

		m_bEasyWinFound = false;

		std::future<int> promise = std::async(&MonteCarloTreeSearch::SearchForEasyWin_, this, game);

		for (int i = 0; i < m_playouts; i++)
		{
			if (m_bEasyWinFound)
				break;
			GomokuGame gameCopy(game);
			m_playoutGameFn(gameCopy);
		}

		int easyIndex = promise.get();
		if (easyIndex > 0 && m_pRoot->GetChildren()[easyIndex])
		{
			m_pRoot->GetChildren()[easyIndex]->SetVisits(INT32_MAX);
			m_pRoot->SetVisits(INT32_MAX);
			return easyIndex;
		}

		return m_pRoot->SelectVisits();
	}

	/*--------------------------------------------------------------*/

	void MonteCarloTreeSearch::DeleteNodes_(int index, MonteCarloNode* pNode)
	{
		pNode->DeleteChildrenExcept(index);
		delete pNode;
	}

	int MonteCarloTreeSearch::SearchForEasyWin_(GomokuGame& game)
	{
		char symbol = game.GetPlayerTurn() ? _P1SYMBOL_ : _P2SYMBOL_;

		char* pCandidates = game.GetCandidateMoves();
		for (int i = 0; i < m_gameSpace; i++)
		{
			if (pCandidates[i] != 0)
			{
				if (game.WillMoveWin(i, symbol))
				{
					m_bEasyWinFound = true;
					return i;
				}
			}
		}

		return -1;
	}

	void MonteCarloTreeSearch::AgentPlayout_(GomokuGame& game)
	{
		if (m_pRoot->GetChildrenCount() > 0)
		{
			int move = m_pRoot->Select(game.GetPlayerTurn());
			MonteCarloNode* pNode = m_pRoot->GetChildren()[move];
			_ASSERT(pNode != nullptr);
			game.PlayMove(move);
			AgentPlayoutAsync_(game, pNode, 1);
			
		}
		else
		{
			GomokuGame gameCopy(game);
			AgentPlayoutAsync_(gameCopy, m_pRoot, 1);
		}
	}

	void MonteCarloTreeSearch::AgentPlayoutAsync_(
		GomokuGame& game,
		MonteCarloNode* pNode,
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

		switch (game.GetGameWinState())
		{
		case WinnerState_enum::P1:
			pNode->RecursiveUpdate(2.0, visitUpdate);
			return;
		case WinnerState_enum::P2:
			pNode->RecursiveUpdate(-2.0, visitUpdate);
			return;
		case WinnerState_enum::Draw:
			pNode->RecursiveUpdate(0.0, visitUpdate);
			return;
		}

		std::promise<torch::Tensor> policyPromise;
		std::promise<double> valuePromise;
		auto policyFuture = policyPromise.get_future();
		auto valueFuture = valuePromise.get_future();
		auto fn = std::function<void(torch::Tensor const&, double)>(
			[&policyPromise, &valuePromise](torch::Tensor const& policy, double value)
		{
			policyPromise.set_value(policy);
			valuePromise.set_value(value);
		});

		m_pAgent->PredictBothAsync(game.GetBoard(), m_gameSpace, game.GetLastMove(), game.GetPlayerTurn(), &fn);
		int size = 0;
		int* pLegalMoves = game.GetLegalMoves(size);

		torch::Tensor const& policy = policyFuture.get();
		auto thread = std::thread(&MonteCarloNode::ExpandChildren, pNode, pLegalMoves, policy, size, game.GetCandidateMoves());
		pNode->RecursiveUpdate(valueFuture.get());
		
		thread.join();
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
			DefaultExpandChildren_(pNode, &game);
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

	void MonteCarloTreeSearch::DefaultExpandChildren_(
		MonteCarloNode* pNode,
		GomokuGame* pGame)
	{
		int size = 0;
		int* pLegalMoves = pGame->GetLegalMoves(size);

		pNode->DefaultExpandChildren(pLegalMoves, size);
	}
}