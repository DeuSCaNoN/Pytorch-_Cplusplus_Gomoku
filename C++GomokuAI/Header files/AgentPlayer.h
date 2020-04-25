#pragma once
#pragma once

#include <IPlayer.h>

#include <GomokuPolicyAgent.h>
#include <MonteCarloTreeSearch.h>

#include <memory>

namespace Player
{
	class AgentPlayer : public IPlayer
	{
	public:
		AgentPlayer(std::shared_ptr<GomokuPolicyAgent> const& pAgent, int rollouts);

		virtual ~AgentPlayer() = default;

		int MakeMove(std::shared_ptr<GomokuGame> pGame, bool bTurn, int& moveToSave);

		void MoveMadeInGame(int moveIndex);

		void ClearTree();

	private:
		std::shared_ptr<GomokuPolicyAgent> m_pAgent;
		std::shared_ptr<MonteCarlo::MonteCarloTreeSearch> m_pTreeSearch;

		int m_lastMoveMade;
	};
}