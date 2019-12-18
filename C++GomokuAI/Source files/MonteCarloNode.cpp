#include "pch.h"
#include "MonteCarloNode.h"

#include <math.h>
namespace MonteCarlo
{

	MonteCarloNode::MonteCarloNode(MonteCarloNode* parent, float prob, int gameSpace)
		: m_visits(0)
		, m_probability(prob)
		, m_qValue(0.0f)
		, m_childrenSize(gameSpace)
		, m_parent(parent)
	{
		m_children = new MonteCarloNode*[m_childrenSize];
		for (int i = 0; i < m_childrenSize; i++)
		{
			m_children[i] = nullptr;
		}
	}

	MonteCarloNode::MonteCarloNode(MonteCarloNode const& other)
	{
		m_parent = other.m_parent;
		m_children = other.m_children;
		m_childrenSize = other.m_childrenSize;
		m_visits = other.m_visits;
		m_probability = other.m_probability;
		m_qValue = other.m_qValue;
	}

	MonteCarloNode::~MonteCarloNode()
	{
		delete m_parent;
		m_parent = nullptr;

		for (int i = 0; i < m_childrenSize; i++)
		{
			delete m_children[i];
		}
		delete m_children;
		m_children = nullptr;
	}

	/*--------------------------------------------------------------*/

	void MonteCarloNode::ExpandChildren(int* actions, float* probs, int size)
	{
		for (int i = 0; i < size; i++)
		{
			int index = actions[i];
			float probability = probs[i];

			if (m_children[index] == nullptr)
			{
				m_children[index] = new MonteCarloNode(this, probability, m_childrenSize);
			}
		}
	}

	double MonteCarloNode::GetValue(short c_puct)
	{
		return m_qValue + ((c_puct * m_probability * sqrt(m_parent->GetVisits())) / (m_visits + 1));
	}

	void MonteCarloNode::Update(float leafValue)
	{
		m_visits++;
		m_qValue += (leafValue - m_qValue) / m_visits;
	}

	void MonteCarloNode::RecursiveUpdate(float leafValue)
	{
		if (m_parent != nullptr)
			m_parent->Update(-leafValue);

		Update(leafValue);
	}

	int MonteCarloNode::GetVisits() const
	{
		return m_visits;
	}

}