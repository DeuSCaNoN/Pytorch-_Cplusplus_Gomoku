#pragma once
#include "MonteCarloNode.h"
#include "GomokuGame.h"

namespace MonteCarlo
{
	class MonteCarloTreeSearch
	{
	public:
		MonteCarloTreeSearch(int gameSpace, int playouts=5000);
		~MonteCarloTreeSearch();

		void StepInTree(int index);

		int GetMove(GomokuGame const& game);
	private:
		void Playout_(GomokuGame& game);

		double AsyncRollout_(GomokuGame* pGame, int index);
		double EvaluateRollout_(GomokuGame& game);

		void ExpandChildrenBluPig_(MonteCarloNode* pNode, GomokuGame* pGame);

		std::function<void(MonteCarloNode*, GomokuGame*)> m_expandChildrenFn;

		MonteCarloNode* m_pRoot;
		int m_playouts;
		int m_gameSpace;
	};
}