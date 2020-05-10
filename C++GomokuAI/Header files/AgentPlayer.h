#pragma once
#pragma once

#include <IPlayer.h>
#include <FwdDecl.h>
#include <GomokuPolicyAgent.h>

#include <memory>

namespace Player
{
	class AgentPlayer : public IPlayer
	{
	public:
		AgentPlayer(std::shared_ptr<GomokuPolicyAgent> const& pAgent, int rollouts);

		virtual ~AgentPlayer() = default;

		int MakeMove(std::shared_ptr<GomokuGame> pGame, bool bTurn, float* moveEstimates);

		void MoveMadeInGame(int moveIndex);

		void ClearTree();

	private:
		std::shared_ptr<GomokuPolicyAgent> m_pAgent;
		std::shared_ptr<MonteCarlo::MonteCarloTreeSearch> m_pTreeSearch;

		int m_lastMoveMade;
	};
}