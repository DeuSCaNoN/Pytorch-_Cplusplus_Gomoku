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
		m_pTreeSearch = std::make_shared<MonteCarlo::MonteCarloTreeSearch>(BOARD_SIDE*BOARD_SIDE, m_pAgent, rollouts);
	}

	int AgentPlayer::MakeMove(std::shared_ptr<GomokuGame> pGame, bool /*bTurn*/, int& moveToSave)
	{
		moveToSave = m_pTreeSearch->GetMove(*pGame);
		return moveToSave;
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