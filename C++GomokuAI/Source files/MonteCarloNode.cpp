#include "pch.h"
#include "MonteCarloNode.h"

#include <math.h>
namespace MonteCarlo
{

	MonteCarloNode::MonteCarloNode(MonteCarloNode* parent, double prob, int gameSpace)
		: m_visits(0)
		, m_probability(prob)
		, m_qValue(0.0f)
		, m_gameSpace(gameSpace)
		, m_childrenSize(0)
		, m_pParent(parent)
	{
		m_ppChildren = new MonteCarloNode*[m_gameSpace];
		for (int i = 0; i < m_gameSpace; i++)
		{
			m_ppChildren[i] = nullptr;
		}
	}

	MonteCarloNode::MonteCarloNode(MonteCarloNode const& other)
	{
		m_pParent = other.m_pParent;
		m_ppChildren = other.m_ppChildren;
		m_childrenSize = other.m_childrenSize;
		m_gameSpace = other.m_gameSpace;
		m_visits = other.m_visits;
		m_probability = other.m_probability;
		m_qValue = other.m_qValue;
	}

	MonteCarloNode::~MonteCarloNode()
	{}

	void MonteCarloNode::DeleteChildrenExcept(unsigned int exception)
	{
		for (int i = 0; i < m_gameSpace; i++)
		{
			if (i != exception)
				delete m_ppChildren[i];
		}
		delete m_ppChildren;
		m_ppChildren = nullptr;
		m_childrenSize = 0;
	}

	/*--------------------------------------------------------------*/

	void MonteCarloNode::ExpandChildren(int* actions, double* probs, int size)
	{
		for (int i = 0; i < size; i++)
		{
			int index = actions[i];
			double probability = probs[i];
			if (!m_ppChildren[index])
			{
				m_ppChildren[index] = new MonteCarloNode(this, probability, m_gameSpace);
				m_childrenSize++;
			}
		}
	}

	double MonteCarloNode::GetValue(short c_puct, bool bPlayerToSearch)
	{
		double value = (c_puct * m_probability * sqrt(log(m_pParent->GetVisits()) / (m_visits + 1)));
		if (!bPlayerToSearch)
			value = -value;
		return (m_qValue / (m_visits + 1)) + value;
	}

	MonteCarloNode** MonteCarloNode::GetChildren()
	{
		return m_ppChildren;
	}

	void MonteCarloNode::Update(double leafValue)
	{
		m_visits++;

		m_qValue += leafValue;
	}

	void MonteCarloNode::RecursiveUpdate(double leafValue)
	{
		if (m_pParent != nullptr)
			m_pParent->RecursiveUpdate(leafValue);

		Update(leafValue);
	}

	int MonteCarloNode::GetVisits() const
	{
		return m_visits;
	}

	int MonteCarloNode::Select(bool playerToCheck, short c_puct)
	{
		int index = -1;
		double maxValue = playerToCheck ? -500.0 : 500.0;
		for (int i = 0; i < m_gameSpace; i++)
		{
			if (m_ppChildren[i] == nullptr)
				continue;

			if (index == -1)
				index = i;
			
			double value = m_ppChildren[i]->GetValue(c_puct, playerToCheck);
			if (playerToCheck)
			{
				if (value > maxValue)
				{
					maxValue = value;
					index = i;
				}
			}
			else
			{
				if (value < maxValue)
				{
					maxValue = value;
					index = i;
				}
			}
		}

		return index;
	}

	int MonteCarloNode::GetChildrenCount() const
	{
		return m_childrenSize;
	}

}