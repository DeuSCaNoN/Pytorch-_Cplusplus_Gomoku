#include "pch.h"
#include <MonteCarloNode.h>

#define NON_CANDIDATE_MOVE -4444.0

namespace MonteCarlo
{

	MonteCarloNode::MonteCarloNode(MonteCarloNode* pParent, double prob, int gameSpace, int moveIndex)
		: m_visits(1)
		, m_probability(prob)
		, m_qValue(0.0f)
		, m_gameSpace(gameSpace)
		, m_childrenSize(0)
		, m_pParent(pParent)
		, bChildrenInitialized(false)
		, m_moveIndexMadeToGetHere(moveIndex)
	{
		m_ppChildren = new MonteCarloNode*[m_gameSpace];
		for (int i = 0; i < m_gameSpace; i++)
		{
			m_ppChildren[i] = nullptr;
		}
	}

	MonteCarloNode::MonteCarloNode(MonteCarloNode const& other)
		: m_moveIndexMadeToGetHere(other.m_moveIndexMadeToGetHere)
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

	void MonteCarloNode::ExpandChildren(int* actions, torch::Tensor const& probs, int size, char* pCandidateMoves)
	{
		torch::Tensor cpuProbs = probs.to(torch::kCPU);

		for (int i = 0; i < size; i++)
		{
			int index = actions[i];

			if (!m_ppChildren[index])
			{
				double probability = -1.0;
				if (pCandidateMoves[index] != 0)
				{
					probability = cpuProbs[0][index].item<float>() * (0.15 / size); // priors probability + Dirichlet noise
				}
				
				m_ppChildren[index] = new MonteCarloNode(this, probability, m_gameSpace, index);
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
				m_ppChildren[index] = new MonteCarloNode(this, probability, m_gameSpace, index);
				m_childrenSize++;
			}
		}
		bChildrenInitialized = true;
	}

	double MonteCarloNode::GetValue(short c_puct, bool bPlayerToSearch)
	{
		if (m_probability < 0)
			return NON_CANDIDATE_MOVE;

		double selectValue = c_puct * m_probability * sqrt(m_pParent->GetVisits()) / m_visits;
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
			if (value == NON_CANDIDATE_MOVE)
				continue;

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

	void MonteCarloNode::SelectBest(bool playerToCheck, int* indicies, short const indexSize, short c_puct)
	{
		double maxValue = playerToCheck ? -1000.0f : DBL_MAX;
		memset(indicies, -1, indexSize * sizeof(int));

		int* pLeastLocation = indicies;

		int itemPushed = 0;
		for (int i = 0; i < m_gameSpace; i++)
		{
			if (m_ppChildren[i] == nullptr)
				continue;

			double currentValue = m_ppChildren[i]->GetValue(c_puct, playerToCheck);
			if (currentValue == NON_CANDIDATE_MOVE)
				continue;

			if (playerToCheck && (currentValue > maxValue) ||
				!playerToCheck && (currentValue < maxValue))
			{
				*pLeastLocation = i;
				maxValue = currentValue;

				if (pLeastLocation == indicies)
					pLeastLocation = &indicies[indexSize - 1];
				else
					pLeastLocation -= 1;

				++itemPushed;
			}
			else if (itemPushed < indexSize)
			{
				*pLeastLocation = i;
				if (pLeastLocation == indicies)
					pLeastLocation = &indicies[indexSize - 1];
				else
					pLeastLocation -= 1;
				++itemPushed;
			}
		}

		if (pLeastLocation != &indicies[indexSize - 1])
		{
			int temp = indicies[0];
			indicies[0] = *(pLeastLocation + 1);
			*(pLeastLocation + 1) = temp;
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