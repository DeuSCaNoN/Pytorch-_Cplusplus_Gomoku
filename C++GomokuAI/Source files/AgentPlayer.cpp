#include "pch.h"

#include <AgentPlayer.h>

#include <MonteCarloTreeSearch.h>
#include <MonteCarloNode.h>
#include <GomokuGame.h>

namespace Player
{
	AgentPlayer::AgentPlayer(std::shared_ptr<GomokuPolicyAgent> const& pAgent, int rollouts)
	{
		m_pAgent = pAgent;
		m_pTreeSearch = std::make_shared<MonteCarlo::MonteCarloTreeSearch>(BOARD_LENGTH, m_pAgent, rollouts);
	}

	void AgentPlayer::UpdateModel(std::string const& modelPath)
	{
		m_pTreeSearch->UpdateModel(modelPath);
	}

	int AgentPlayer::MakeMove(std::shared_ptr<GomokuGame> pGame, bool bTurn, float* pMoveEstimates)
	{
//		auto start = std::chrono::steady_clock::now();
		int move = m_pTreeSearch->GetMove(*pGame);
//		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
//		std::cout << duration.count() << std::endl;

		if (pMoveEstimates)
		{
			memset(pMoveEstimates, 0, BOARD_LENGTH * sizeof(float));
			MonteCarlo::MonteCarloNode** ppChildren = m_pTreeSearch->GetRoot()->GetChildren();
			for (int i = 0; i < BOARD_LENGTH; i++)
			{
				if (ppChildren[i] != nullptr)
				{
					pMoveEstimates[i] = (float)ppChildren[i]->GetVisits() / m_pTreeSearch->GetRoot()->GetVisits();
				}
			}
		}
		
		return move;
	}

	void AgentPlayer::ClearTree()
	{
		m_pTreeSearch->Reset();
	}

	std::string AgentPlayer::PrintWinningStatement()
	{
		return "Agent won";
	}
}