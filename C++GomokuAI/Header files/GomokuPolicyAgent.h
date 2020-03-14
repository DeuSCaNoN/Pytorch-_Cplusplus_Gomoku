#pragma once

#include "torch/torch.h"

#include <vector>
#include <string>

#define BOARD_SIDE 15

// Define a new Module.
struct Net : torch::nn::Module {
	Net() {
		// Construct and register two Linear submodules.
		convN1 = register_module("convN1", torch::nn::Conv2d(4, 32, 3));
		convN2 = register_module("convN2", torch::nn::Conv2d(32, 64, 3));
		convN3 = register_module("convN3", torch::nn::Conv2d(64, 128, 3));

		policyN1 = register_module("policyN1", torch::nn::Conv2d(128, 4, 1));
		policyN2 = register_module("policyN2", torch::nn::Linear(4 * 9 * 9, BOARD_SIDE*BOARD_SIDE));

		valueN1 = register_module("valueN1", torch::nn::Conv2d(128, 2, 1));
		valueN2 = register_module("valueN2", torch::nn::Linear(2 * 9 * 9, 64));
		valueN3 = register_module("valueN3", torch::nn::Linear(64, 1));
	}

	torch::Tensor forwadPolicy(torch::Tensor x)
	{
		x = torch::relu(convN1->forward(x));
		x = torch::relu(convN2->forward(x));
		x = torch::relu(convN3->forward(x));

		x = torch::relu(policyN1->forward(x));
		return torch::log_softmax(policyN2->forward(x.reshape({x.sizes()[0], 4 *9 * 9})), 1);
	}

	torch::Tensor forwadValue(torch::Tensor x)
	{
		x = torch::relu(convN1->forward(x));
		x = torch::relu(convN2->forward(x));
		x = torch::relu(convN3->forward(x));

		x = torch::relu(valueN1->forward(x));
		x = torch::relu(valueN2->forward(x.reshape({x.sizes()[0], 2 * 9 * 9 })));
		return torch::tanh(valueN3->forward(x));
	}

	// Use one of many "standard library" modules.
	torch::nn::Conv2d convN1{ nullptr }, convN2{ nullptr }, convN3{ nullptr }, policyN1{ nullptr }, valueN1{ nullptr };
	torch::nn::Linear policyN2{ nullptr }, valueN2{ nullptr }, valueN3{ nullptr };
};

class GomokuPolicyAgent
{
public:
	GomokuPolicyAgent(std::string const& modelPath = "GomokuModel.pt");

	~GomokuPolicyAgent();

	void SaveModel();

	void ReloadCpuModel();

	double PredictValue(char* board, int size, int lastMoveIndex, bool bTurn);

	std::vector<double> PredictMove(char* board, int size, int lastMoveIndex, bool bTurn);
private:

	torch::Tensor CreateTensorBoard_(char* board, int size, int lastMoveIndex, bool bTurn);

	std::shared_ptr<Net> m_pNetworkCpu;
	std::shared_ptr<Net> m_pNetworkGpu;

	std::string m_modelPath;
};