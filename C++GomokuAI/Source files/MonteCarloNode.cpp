#include "pch.h"
#include <MonteCarloNode.h>

#define NON_CANDIDATE_MOVE -4444.0

namespace MonteCarlo
{

	MonteCarloNode::MonteCarloNode(MonteCarloNode* pParent, double prob, int gameSpace)
		: m_visits(1)
		, m_probability(prob)
		, m_qValue(0.0f)
		, m_gameSpace(gameSpace)
		, m_childrenSize(0)
		, m_pParent(pParent)
		, bChildrenInitialized(false)
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
		if (bChildrenInitialized)
		{
			for (int i = 0; i < m_gameSpace; i++)
			{
				if (i != exception && m_ppChildren[i] != nullptr)
				{
					m_ppChildren[i]->DeleteChildrenExcept(-1);
					delete m_ppChildren[i];
				}
				else if (i == exception)
				{
					m_ppChildren[i]->UnsetParent_();
				}
			}
		}
		delete[] m_ppChildren;
		m_ppChildren = nullptr;
		m_childrenSize = 0;
	}

	/*--------------------------------------------------------------*/

	void MonteCarloNode::ExpandChildren(int* actions, torch::Tensor& probs, int size)
	{
		probs = probs.to(torch::kCPU);
		for (int i = 0; i < size; i++)
		{
			int index = actions[i];
			double probability = probs[0][index].item<float>() * (0.15 / size); // priors probability + Dirichlet noise
			if (!m_ppChildren[index])
			{
				m_ppChildren[index] = new MonteCarloNode(this, probability, m_gameSpace);
				m_childrenSize++;
			}
		}
		bChildrenInitialized = true;
	}

	void MonteCarloNode::DefaultExpandChildren(int* actions, int size)
	{
		for (int i = 0; i < size; i++)
		{
			int index = actions[i];
			double probability = 1 / size;
			if (!m_ppChildren[index])
			{
				m_ppChildren[index] = new MonteCarloNode(this, probability, m_gameSpace);
				m_childrenSize++;
			}
		}
		bChildrenInitialized = true;
	}

	double MonteCarloNode::GetValue(short c_puct, bool bPlayerToSearch)
	{
		float selectValue = c_puct * m_probability * sqrt(m_pParent->GetVisits()) / m_visits;
		if (!bPlayerToSearch)
			selectValue = -selectValue;
		return (m_qValue / m_visits) + selectValue;
	}

	MonteCarloNode** MonteCarloNode::GetChildren()
	{
		return m_ppChildren;
	}

	void MonteCarloNode::Update(double leafValue, int visitUpdate)
	{
		m_visits += visitUpdate;
		m_qValue += leafValue;
	}

	void MonteCarloNode::RecursiveUpdate(double leafValue, int visitUpdate)
	{
		if (m_pParent != nullptr)
			m_pParent->RecursiveUpdate(leafValue, visitUpdate);

		Update(leafValue, visitUpdate);
	}

	int MonteCarloNode::GetVisits() const
	{
		return m_visits;
	}

	int MonteCarloNode::Select(bool playerToCheck, short c_puct)
	{
		int index = -1;
		double maxValue = playerToCheck ? -500 : DBL_MAX;
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

	int MonteCarloNode::SelectVisits() const
	{
		int index;
		int maxVisits = -1;
		for (int i = 0; i < m_gameSpace; i++)
		{
			if (m_ppChildren[i] == nullptr)
				continue;

			int childVisits = m_ppChildren[i]->GetVisits();
			if (childVisits > maxVisits)
			{
				maxVisits = childVisits;
				index = i;
			}
		}

		return index;
	}

	void MonteCarloNode::SelectBestFour(bool playerToCheck, int* indicies, short c_puct)
	{
		float maxValue = playerToCheck ? -1000.0f : DBL_MAX;
		indicies[0] = -1;
		indicies[1] = -1;
		indicies[2] = -1;
		indicies[3] = -1;

		int offset = m_gameSpace / 3;

		int* pIndex1 = indicies;
		int* pIndex2 = &indicies[1];
		int* pIndex3 = &indicies[2];
		int* pIndex4 = &indicies[3];
		int itemPushed = 0;
		for (int i = 0; i < m_gameSpace; i++)
		{
			int indexToTest = (i + offset) % m_gameSpace;
			if (m_ppChildren[indexToTest] == nullptr)
				continue;

			float currentValue = m_ppChildren[indexToTest]->GetValue(c_puct, playerToCheck);
			if (playerToCheck && (currentValue > maxValue) ||
				!playerToCheck && (currentValue < maxValue))
			{
				int* pTemp = pIndex4;
				pIndex4 = pIndex3;
				pIndex3 = pIndex2;
				pIndex2 = pIndex1;
				pIndex1 = pTemp;
				*pIndex1 = indexToTest;
				maxValue = currentValue;

				++itemPushed;
			}
			else if (itemPushed < 4)
			{
				switch (itemPushed)
				{
				case 0:
					*pIndex1 = indexToTest;
					break;
				case 1:
					*pIndex2 = indexToTest;
					break;
				case 2:
					*pIndex3 = indexToTest;
					break;
				case 3:
					*pIndex4 = indexToTest;
					break;
				default:
					break;
				}
				++itemPushed;
			}
		}

		if (pIndex1 != indicies)
		{
			int temp = indicies[0];
			indicies[0] = *pIndex1;
			*pIndex1 = temp;
		}
	}

	int MonteCarloNode::GetChildrenCount() const
	{
		return m_childrenSize;
	}

	void MonteCarloNode::SetVisits(int newVisists)
	{
		m_visits = newVisists;
	}

	void MonteCarloNode::UnsetParent_()
	{
		m_pParent = nullptr;
	}

	double const MonteCarloNode::GetProbability() const
	{
		return m_probability;
	}
}