#pragma once

namespace MonteCarlo
{
	class MonteCarloNode
	{
	public:
		MonteCarloNode(MonteCarloNode* parent, double prob, int gameSpace);
		MonteCarloNode(MonteCarloNode const& other);
		~MonteCarloNode();
		// CALL THIS WHENEVER DELETING MONTECARLO TREE NODES
		void DeleteChildrenExcept(unsigned int exception); // pass -1 to delete all children

		void ExpandChildren(int* actions, double* probs, int size);
		double GetValue(short c_puct, bool bPlayerToSearch);
		void Update(double leafValue);
		int Select(bool playerToCheck, short c_puct=5);

		MonteCarloNode** GetChildren();

		void RecursiveUpdate(double leafValue);
		int GetVisits() const;

		int GetChildrenCount() const;
	private:
		MonteCarloNode* m_pParent;
		MonteCarloNode** m_ppChildren;
		int m_gameSpace;
		int m_childrenSize;

		int m_visits;
		double m_probability;
		double m_qValue;
	};
}