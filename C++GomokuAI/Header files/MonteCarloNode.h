#pragma once

#include <vector>

namespace MonteCarlo
{
	class MonteCarloNode
	{
	public:
		MonteCarloNode(MonteCarloNode* parent, double prob, int gameSpace, int moveIndex = -1);
		MonteCarloNode(MonteCarloNode const& other);
		~MonteCarloNode();
		// CALL THIS WHENEVER DELETING MONTECARLO TREE NODES
		void DeleteChildrenExcept(unsigned int exception); // pass -1 to delete all children

		void ExpandChildren(int* actions, torch::Tensor const& probs, int size, char* pCandidateMoves);
		void DefaultExpandChildren(int* actions, int size);
		double GetValue(short c_puct, bool bPlayerToSearch);
		int Select(bool playerToCheck, short c_puct = 5);

		int SelectVisits() const;

		double const GetProbability() const;

		void SelectBest(bool playerToCheck, int* indicies, short const indexSize, short c_puct = 5);

		MonteCarloNode** GetChildren();

		void Update(double leafValue, int visitUpdate = 1);
		void RecursiveUpdate(double leafValue, int visitUpdate = 1);
		int GetVisits() const;

		int GetChildrenCount() const;

		void SetVisits(int newVisists);
	private:

		void UnsetParent_();

		MonteCarloNode* m_pParent;
		MonteCarloNode** m_ppChildren;
		int m_gameSpace;
		int m_childrenSize;

		bool bChildrenInitialized;

		int m_visits;
		double m_probability;
		double m_qValue;

		int const m_moveIndexMadeToGetHere;
	};
}