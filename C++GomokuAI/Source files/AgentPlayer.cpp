#include "pch.h"

#include <MonteCarloNode.h>

#include <AgentPlayer.h>
#include <GomokuUtils.h>

namespace Player
{
	AgentPlayer::AgentPlayer(std::shared_ptr<GomokuPolicyAgent> const& pAgent, int rollouts)
		: m_lastMoveMade(-1)
	{
		m_pAgent = pAgent;
		m_pTreeSearch = std::make_shared<MonteCarlo::MonteCarloTreeSearch>(225, m_pAgent, rollouts);
	}

	int AgentPlayer::MakeMove(std::shared_ptr<GomokuGame> pGame, bool /*bTurn*/, int& moveToSave)
	{
		int moveMade =  m_pTreeSearch->GetMove(*pGame);
		moveToSave = moveMade;

/*		double maxProbability = -1.0;
		for (int i = 0; i < 225; ++i)
		{
			MonteCarlo::MonteCarloNode* pChild = pRoot->GetChildren()[i];
			if (pChild != nullptr && pChild->GetProbability() > maxProbability)
			{
				maxProbability = pChild->GetProbability();
				moveToSave = i;
			}

		}*/
		return moveMade;
	}

	void AgentPlayer::MoveMadeInGame(int moveIndex)
	{
		if (moveIndex != m_lastMoveMade)
		{
			m_pTreeSearch->StepInTree(moveIndex);
			m_lastMoveMade = moveIndex;
		}
	}

	void AgentPlayer::ClearTree()
	{
		m_pTreeSearch->Reset();
	}
}