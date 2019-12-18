#pragma once

namespace MonteCarlo
{
	class MonteCarloNode
	{
	public:
		MonteCarloNode(MonteCarloNode* parent, float prob, int gameSpace);
		MonteCarloNode(MonteCarloNode const& other);
		~MonteCarloNode();

		void ExpandChildren(int* actions, float* probs, int size);
		double GetValue(short c_puct);
		void Update(float leafValue);

		void RecursiveUpdate(float leafValue);

		int GetVisits() const;
	private:
		MonteCarloNode* m_parent;
		MonteCarloNode** m_children;
		int m_childrenSize;

		int m_visits;
		float m_probability;
		float m_qValue;
	};
}