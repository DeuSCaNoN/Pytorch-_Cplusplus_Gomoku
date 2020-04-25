#pragma once

#include <torch/torch.h>

#include <vector>

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

		void ExpandChildren(int* actions, torch::Tensor& probs, int size);
		void DefaultExpandChildren(int* actions, int size);
		double GetValue(short c_puct, bool bPlayerToSearch);
		void Update(double leafValue);
		int Select(bool playerToCheck, short c_puct=100);

		int SelectVisits() const;

		double const GetProbability() const;

		void SelectBestFour(bool playerToCheck, int* indicies, short c_puct = 100);

		MonteCarloNode** GetChildren();

		void RecursiveUpdate(double leafValue);
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
	};
}