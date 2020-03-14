#include "pch.h"

#include "GomokuPolicyAgent.h"
#include "GomokuUtils.h"

#include <fstream>
#include <exception>

/*--------------------------------------------------------------*/

GomokuPolicyAgent::GomokuPolicyAgent(std::string const& modelPath)
{
	auto test = torch::nn::Linear(25, 1);
	torch::Tensor tensor = torch::randn({ 25 });
	test->forward(tensor);
	m_modelPath = modelPath;
	std::ifstream f(modelPath.c_str());
	if (f.good())
	{
		torch::load(m_pNetworkCpu, modelPath);
		torch::load(m_pNetworkGpu, modelPath);
	}
	else
	{
		m_pNetworkCpu = std::make_shared<Net>();
		m_pNetworkGpu = m_pNetworkCpu;
	}

	m_pNetworkGpu->to(torch::kCUDA);
}

GomokuPolicyAgent::~GomokuPolicyAgent()
{
}

/*--------------------------------------------------------------*/

void GomokuPolicyAgent::SaveModel()
{
	torch::save(m_pNetworkGpu, m_modelPath);
}

void GomokuPolicyAgent::ReloadCpuModel()
{
	std::ifstream f(m_modelPath.c_str());
	if (f.good)
	{
		torch::load(m_pNetworkCpu, m_modelPath);
	}
	else
	{
		std::cout << "Cpu model failed to reload, could not find " << m_modelPath;
	}
}

double GomokuPolicyAgent::PredictValue(char* board, int size, int lastMoveIndex, bool bTurn)
{
	torch::Tensor boardTensor = CreateTensorBoard_(board, size, lastMoveIndex, bTurn);
	torch::Tensor valueTensor = m_pNetworkCpu->forwadValue(boardTensor.reshape({1,4,BOARD_SIDE,BOARD_SIDE}));
	return valueTensor.item<double>();
}

std::vector<double> GomokuPolicyAgent::PredictMove(char* board, int size, int lastMoveIndex, bool bTurn)
{
	torch::Tensor boardTensor = CreateTensorBoard_(board, size, lastMoveIndex, bTurn);
	torch::Tensor policyTensor = m_pNetworkCpu->forwadPolicy(boardTensor.reshape({ 1,4,BOARD_SIDE,BOARD_SIDE }));
	std::vector<double> boardPolicyVect(BOARD_SIDE*BOARD_SIDE);

	for (int i = 0; i < BOARD_SIDE*BOARD_SIDE; i++)
	{
		boardPolicyVect[i] = policyTensor[0][i].item<float>();
	}

	return boardPolicyVect;
}

/*--------------------------------------------------------------*/

torch::Tensor GomokuPolicyAgent::CreateTensorBoard_(char* board, int size, int lastMoveIndex, bool bTurn)
{
	char turnSymbol = bTurn ? 1 : 2;
	torch::Tensor finalTensor = torch::zeros({ 4, BOARD_SIDE, BOARD_SIDE });
	auto accessor = finalTensor.accessor<float, 3>();
	
	short row, col;
	if (lastMoveIndex >= 0)
	{
		GomokuUtils::ConvertToRowCol(lastMoveIndex, BOARD_SIDE, row, col);
		accessor[2][row][col] = 1.0;
	}
	
	for (int i = 0; i < size; ++i)
	{
		GomokuUtils::ConvertToRowCol(i, BOARD_SIDE, row, col);
		if (board[i] == turnSymbol)
		{
			accessor[0][row][col] = 1.0;
		}
		else if (board[i] != 0)
		{
			accessor[1][row][col] = 1.0;
		}

		if (bTurn)
		{
			accessor[3][row][col] = 1.0;
		}
	}
	return finalTensor;
}