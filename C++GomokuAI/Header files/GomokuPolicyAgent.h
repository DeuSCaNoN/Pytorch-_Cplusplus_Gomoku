#pragma once

#include "torch/torch.h"
#include <FwdDecl.h>

#include <vector>
#include <string>

struct TrainingExample
{
	char board[BOARD_LENGTH];
	float pMoveEstimate[BOARD_LENGTH];
	float boardValue;
	int lastMove;
};

// Define a new Module.
struct Net : torch::nn::Module
{
	Net(short resNetSize = 5);

	torch::Tensor forwadPolicy(torch::Tensor x);

	torch::Tensor forwadValue(torch::Tensor x);

	void forwardBoth(torch::Tensor x, torch::Tensor& policy, torch::Tensor& value);

	void train(bool on = true);

	short m_residualNetSize;

	// Use one of many "standard library" modules.
	torch::nn::Conv2d convN1{ nullptr }, policyN1{ nullptr }, valueN1{ nullptr };
	torch::nn::Linear policyN2{ nullptr }, valueN2{ nullptr }, valueN3{ nullptr };

	std::vector<torch::nn::BatchNorm2d> m_residualBatch1;
	std::vector<torch::nn::BatchNorm2d> m_residualBatch2;
	std::vector<torch::nn::Conv2d> m_residualConv1;
	std::vector<torch::nn::Conv2d> m_residualConv2;

	torch::nn::BatchNorm2d batchNorm1{nullptr}, batchNorm2{ nullptr }, batchNorm3{ nullptr };

private:
	void Ctor_();
};

class GomokuPolicyAgent
{
public:
	GomokuPolicyAgent(std::string const& modelPath = "GomokuModel" + std::to_string(BOARD_SIDE) + "_" + std::to_string(BOARD_WIN) + ".pt");

	~GomokuPolicyAgent();

	void SaveModel();

	void Train(std::vector<TrainingExample>& trainingExamples, double learningRate, unsigned int epoch = 10);

	double PredictValue(char* board, int size, int lastMoveIndex, bool bTurn);

	torch::Tensor PredictMove(char* board, int size, int lastMoveIndex, bool bTurn);

	std::string const& GetModelPath() const;
private:

	torch::Tensor CreateTensorBoard_(char* board, int size, int lastMoveIndex, bool bTurn);

	float TrainGpuAsync_(
		torch::optim::Optimizer& adamOptimizer,
		torch::Tensor inputTensor,
		torch::Tensor valueAnswerTensor,
		torch::Tensor policyAnswerTensor);

	std::shared_ptr<Net> m_pNetworkGpu;

	std::string m_modelPath;
};