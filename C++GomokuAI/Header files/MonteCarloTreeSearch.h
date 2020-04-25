#pragma once
#include "MonteCarloNode.h"
#include "GomokuGame.h"
#include "GomokuPolicyAgent.h"

namespace MonteCarlo
{
	class MonteCarloTreeSearch
	{
	public:
		MonteCarloTreeSearch(int gameSpace, std::shared_ptr<GomokuPolicyAgent> pAgent, int playouts=5000);
		~MonteCarloTreeSearch();

		void Reset();
		void StepInTree(int index);

		MonteCarloNode* const GetRoot();

		int GetMove(GomokuGame const& game);
	private:
		void AgentPlayout_(GomokuGame& game);
		void AgentPlayoutAsync_(GomokuGame& game, MonteCarloNode* pNode);
		void DefaultPlayout_(GomokuGame& game);

		double AsyncRollout_(GomokuGame* pGame, int index);
		double DefaultEvaluateRollout_(GomokuGame const& game);

		void ExpandChildrenPolicyAgent_(MonteCarloNode* pNode, GomokuGame* pGame);
		void DefaultExpandChildren_(MonteCarloNode* pNode, GomokuGame* pGame);

		std::function<void(MonteCarloNode*, GomokuGame*)> m_expandChildrenFn;
		std::function<void(GomokuGame&)> m_playoutGameFn;

		MonteCarloNode* m_pRoot;
		int m_playouts;
		int m_gameSpace;

		std::shared_ptr<GomokuPolicyAgent> m_pAgent;
	};
}