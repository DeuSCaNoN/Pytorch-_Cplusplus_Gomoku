#include "pch.h"
#include "MonteCarloTreeSearch.h"
#include "Utils.h"

#include <stdlib.h>
#include <thread>
#include <future>
#include <vector>

namespace MonteCarlo
{
	MonteCarloTreeSearch::MonteCarloTreeSearch(int gameSpace, int playouts)
		: m_playouts(playouts)
		, m_gameSpace(gameSpace)
		, m_pRoot(new MonteCarloNode(nullptr, 1.0, gameSpace))
	{}

	MonteCarloTreeSearch::~MonteCarloTreeSearch()
	{
		m_pRoot->DeleteChildrenExcept(-1);
		delete m_pRoot;
	}

	/*--------------------------------------------------------------*/

	void MonteCarloTreeSearch::StepInTree(int index)
	{
		_ASSERT(index >= 0 && index < m_gameSpace);
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
		for (int i = 0; i < m_playouts; i++)
		{
			GomokuGame gameCopy(game);
			Playout_(gameCopy);
		}

		return m_pRoot->Select(game.GetPlayerTurn());
	}

	/*--------------------------------------------------------------*/

	void MonteCarloTreeSearch::Playout_(GomokuGame& game)
	{
		MonteCarloNode* pNode = m_pRoot;
		bool playerToSearch = game.GetPlayerTurn();
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
			int size = 0;
			int* pLegalMoves = game.GetLegalMoves(size);

			double* probs = new double[size];
			for (int i = 0; i < size; i++)
			{
				probs[i] = (double)1 / size;
			}
			pNode->ExpandChildren(pLegalMoves, probs, size);
			delete probs;
		}

		double leafValue = EvaluateRollout_(game);

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

	double MonteCarloTreeSearch::EvaluateRollout_(GomokuGame& game)
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
}