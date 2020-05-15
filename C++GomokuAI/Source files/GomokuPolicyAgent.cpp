#include "pch.h"

#include "GomokuPolicyAgent.h"
#include "GomokuUtils.h"

#include "THC/THCCachingHostAllocator.h"
#include <fstream>
#include <future>
#include <exception>

#define BATCH_SIZE 64U
#define BATCH_VERBOSE_SIZE 3200 // BATCH_SIZE * 50
#define CONV_2D_FILTER_SIZE 256

/*--------------------------------------------------------------*/

void Net::Ctor_()
{
	// Construct and register two Linear submodules.
	convN1 = register_module("convN1", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, CONV_2D_FILTER_SIZE, 3).padding(1)));
	batchNorm1 = register_module("batchNorm1", torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(CONV_2D_FILTER_SIZE)));

	m_residualBatch1.reserve(m_residualNetSize);
	m_residualBatch2.reserve(m_residualNetSize);
	m_residualConv1.reserve(m_residualNetSize);
	m_residualConv2.reserve(m_residualNetSize);
	for (short i = 0; i < m_residualNetSize; i++)
	{
		std::string batch1Name = "resNetBatch1" + std::to_string(i);
		std::string batch2Name = "resNetBatch2" + std::to_string(i);
		std::string conv1Name = "resNetConv1" + std::to_string(i);
		std::string conv2Name = "resNetConv2" + std::to_string(i);
		m_residualBatch1.push_back(register_module(batch1Name, torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(CONV_2D_FILTER_SIZE))));
		m_residualBatch2.push_back(register_module(batch2Name, torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(CONV_2D_FILTER_SIZE))));
		m_residualConv1.push_back(register_module(conv1Name, torch::nn::Conv2d(torch::nn::Conv2dOptions(CONV_2D_FILTER_SIZE, CONV_2D_FILTER_SIZE, 3).padding(1))));
		m_residualConv2.push_back(register_module(conv2Name, torch::nn::Conv2d(torch::nn::Conv2dOptions(CONV_2D_FILTER_SIZE, CONV_2D_FILTER_SIZE, 3).padding(1))));
	}

	policyN1 = register_module("policyN1", torch::nn::Conv2d(CONV_2D_FILTER_SIZE, 4, 1));
	batchNorm2 = register_module("batchNorm2", torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(4)));
	policyN2 = register_module("policyN2", torch::nn::Linear(4 * BOARD_LENGTH, BOARD_LENGTH));

	valueN1 = register_module("valueN1", torch::nn::Conv2d(CONV_2D_FILTER_SIZE, 2, 1));
	batchNorm3 = register_module("batchNorm3", torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(2)));
	valueN2 = register_module("valueN2", torch::nn::Linear(2 * BOARD_LENGTH, 64));
	valueN3 = register_module("valueN3", torch::nn::Linear(64, 1));
}

Net::Net(short resNetSize)
	: m_residualNetSize(resNetSize)
{
	Ctor_();
}

torch::Tensor Net::forwadPolicy(torch::Tensor x)
{
	x = torch::relu(batchNorm1->forward(convN1->forward(x)));
	for (int i = 0; i < m_residualNetSize; i++)
	{
		torch::Tensor residualCopy = x.clone();
		x = torch::relu(m_residualBatch1[i]->forward(m_residualConv1[i]->forward(x)));
		x = m_residualBatch2[i]->forward(m_residualConv2[i]->forward(x));
		x += residualCopy;
		x = torch::relu(x);
	}

	x = torch::relu(batchNorm2->forward(policyN1->forward(x)));
	return torch::log_softmax(policyN2->forward(x.reshape({ x.sizes()[0], 4 * BOARD_LENGTH })), 1);
}

torch::Tensor Net::forwadValue(torch::Tensor x)
{
	x = torch::relu(batchNorm1->forward(convN1->forward(x)));
	for (int i = 0; i < m_residualNetSize; i++)
	{
		torch::Tensor residualCopy = x.clone();
		x = torch::relu(m_residualBatch1[i]->forward(m_residualConv1[i]->forward(x)));
		x = m_residualBatch2[i]->forward(m_residualConv2[i]->forward(x));
		x += residualCopy;
		x = torch::relu(x);
	}

	x = torch::relu(batchNorm3->forward(valueN1->forward(x)));
	x = torch::relu(valueN2->forward(x.reshape({ x.sizes()[0], 2 * BOARD_LENGTH })));
	return torch::tanh(valueN3->forward(x));
}

void Net::forwardBoth(torch::Tensor x, torch::Tensor& policy, torch::Tensor& value)
{
	x = torch::relu(batchNorm1->forward(convN1->forward(x)));
	for (int i = 0; i < m_residualNetSize; i++)
	{
		torch::Tensor residualCopy = x.clone();
		x = torch::relu(m_residualBatch1[i]->forward(m_residualConv1[i]->forward(x)));
		x = m_residualBatch2[i]->forward(m_residualConv2[i]->forward(x));
		x += residualCopy;
		x = torch::relu(x);
	}

	policy = torch::relu(batchNorm2->forward(policyN1->forward(x)));
	policy = torch::log_softmax(policyN2->forward(policy.reshape({ policy.sizes()[0], 4 * BOARD_LENGTH })), 1);

	value = torch::relu(batchNorm3->forward(valueN1->forward(x)));
	value = torch::relu(valueN2->forward(value.reshape({ value.sizes()[0], 2 * BOARD_LENGTH })));
	value = torch::tanh(valueN3->forward(value));
}

void Net::train(bool on)
{
	for (int i = 0; i < m_residualNetSize; i++)
	{
		m_residualBatch1[i]->train(on);
		m_residualBatch2[i]->train(on);
	}
	batchNorm1->train(on);
	batchNorm2->train(on);
	batchNorm3->train(on);
}

/*--------------------------------------------------------------*/

GomokuPolicyAgent::GomokuPolicyAgent(std::string const& modelPath)
{
	m_pNetworkGpu = std::make_shared<Net>();

	m_modelPath = modelPath;
	std::ifstream f(modelPath);
	if (f.good())
	{	
		torch::load(m_pNetworkGpu, modelPath);
	}

	m_pNetworkGpu->to(torch::kCUDA);
	m_pNetworkGpu->eval();
}

GomokuPolicyAgent::~GomokuPolicyAgent()
{
	THCCachingHostAllocator_emptyCache();
}

/*--------------------------------------------------------------*/

void GomokuPolicyAgent::SaveModel()
{
	torch::save(m_pNetworkGpu, m_modelPath);
	std::cout << "Model Saved" << std::endl;
}

double GomokuPolicyAgent::PredictValue(char* board, int size, int lastMoveIndex, bool bTurn)
{
	torch::Tensor boardTensor = CreateTensorBoard_(board, size, lastMoveIndex, bTurn).reshape({ 1,4,BOARD_SIDE,BOARD_SIDE });
	torch::Tensor valueTensor = m_pNetworkGpu->forwadValue(boardTensor.to(torch::kCUDA));
	return valueTensor[0].item<double>();
}

torch::Tensor GomokuPolicyAgent::PredictMove(char* board, int size, int lastMoveIndex, bool bTurn)
{
	torch::Tensor boardTensor = CreateTensorBoard_(board, size, lastMoveIndex, bTurn).reshape({ 1,4,BOARD_SIDE,BOARD_SIDE }).to(torch::kCUDA);
	torch::Tensor policyTensor = m_pNetworkGpu->forwadPolicy(boardTensor).exp();

	return std::move(policyTensor);
}

void GomokuPolicyAgent::Train(std::vector<TrainingExample>& trainingExamples, double learningRate, unsigned int epoch)
{
	m_pNetworkGpu->train();
	long long setSize = trainingExamples.size();
	std::future<float> promise;

	bool bRepopulate = trainingExamples.size() > BATCH_SIZE;
	torch::Tensor inputTensor;
	torch::Tensor policyAnswerTensor;
	torch::Tensor valueAnswerTensor;
	
	float lossAggregate = 0.0f;
	for (unsigned int i = 0; i < epoch; i++)
	{
		torch::optim::SGD optimizer(m_pNetworkGpu->parameters(), learningRate);
		lossAggregate = 0.0f;
		std::cout << "starting epoch " << i << std::endl;

		if (bRepopulate || i == 0)
		{
			long long batchSize = setSize >= BATCH_SIZE ? BATCH_SIZE : setSize;
			inputTensor = torch::zeros({ batchSize, 4, BOARD_SIDE, BOARD_SIDE });
			policyAnswerTensor = torch::zeros({ batchSize, BOARD_LENGTH });
			valueAnswerTensor = torch::zeros({ batchSize, 1 });
			size_t index = 0;
			int batchIndex = 0;
			
			for (TrainingExample& example : trainingExamples)
			{
				if (example.lastMove < -1)
				{
					std::cout << "ERROR LAST MOVE WASN'T MADE";
					++index;
					continue;
				}

				bool bTurn = example.board[example.lastMove] == 2;
				inputTensor[batchIndex] = CreateTensorBoard_(
					example.board,
					BOARD_LENGTH,
					example.lastMove,
					bTurn);

				valueAnswerTensor[batchIndex] = example.boardValue;
				for (int i = 0; i < BOARD_LENGTH; i++)
					policyAnswerTensor[batchIndex][i] = example.pMoveEstimate[i];

				++batchIndex;
				++index;

				if (batchIndex >= BATCH_SIZE || index == setSize)
				{
					if (promise.valid())
						lossAggregate += promise.get();

					promise = std::async(
						&GomokuPolicyAgent::TrainGpuAsync_,
						this,
						std::move(optimizer),
						inputTensor,
						valueAnswerTensor,
						policyAnswerTensor);

					batchSize = setSize - index;
					if (batchSize < BATCH_SIZE && batchSize > 0)
					{
						inputTensor = torch::zeros({ batchSize, 4, BOARD_SIDE, BOARD_SIDE });
						policyAnswerTensor = torch::zeros({ batchSize, BOARD_LENGTH });
						valueAnswerTensor = torch::zeros({ batchSize, 1 });
					}

					batchIndex = 0;
				}
			}

			if (promise.valid())
			{
				lossAggregate += promise.get();
				std::cout << lossAggregate / (1+(trainingExamples.size() / BATCH_SIZE)) << std::endl;
			}
		}
		else
		{
			std::cout << TrainGpuAsync_(optimizer, inputTensor, valueAnswerTensor, policyAnswerTensor) << std::endl;
		}
	}

	if (promise.valid())
	{
		lossAggregate += promise.get();
		std::cout << lossAggregate / trainingExamples.size() << std::endl;
	}

	m_pNetworkGpu->eval();
}

std::string const& GomokuPolicyAgent::GetModelPath() const
{
	return m_modelPath;
}

/*--------------------------------------------------------------*/

torch::Tensor GomokuPolicyAgent::CreateTensorBoard_(char* board, int size, int lastMoveIndex, bool bTurn)
{
	torch::Tensor finalTensor = torch::zeros({ 4, BOARD_SIDE, BOARD_SIDE });
	
	short row, col;
	if (lastMoveIndex >= 0)
	{
		GomokuUtils::ConvertToRowCol(lastMoveIndex, BOARD_SIDE, row, col);
		finalTensor[2][row][col] = 1.0;
	}
	
	for (int i = 0; i < size; ++i)
	{
		GomokuUtils::ConvertToRowCol(i, BOARD_SIDE, row, col);
		if (board[i] == 1)
		{
			finalTensor[0][row][col] = 1.0;
		}
		else if (board[i] != 0)
		{
			finalTensor[1][row][col] = 1.0;
		}

		if (bTurn)
		{
			finalTensor[3][row][col] = 1.0;
		}
	}

	return finalTensor;
}

float GomokuPolicyAgent::TrainGpuAsync_(
	torch::optim::Optimizer& optimizer,
	torch::Tensor inputTensor,
	torch::Tensor valueAnswerTensor,
	torch::Tensor policyAnswerTensor)
{
	optimizer.zero_grad();
	torch::Tensor policyTensor;
	torch::Tensor valueTensor;

	m_pNetworkGpu->forwardBoth(inputTensor.to(torch::kCUDA),
		policyTensor,
		valueTensor);

//	std::cout << valueTensor << std::endl;
	torch::Tensor valueLoss = (valueTensor - valueAnswerTensor.to(torch::kCUDA)).pow(2); // test new loss equation
	torch::Tensor policyLoss = torch::sum(-policyAnswerTensor.to(torch::kCUDA) * policyTensor, 1);

	torch::Tensor lossTensor = (valueLoss + policyLoss).mean();

	float loss = lossTensor.item<float>();
	lossTensor.backward();
	optimizer.step();

	return lossTensor.item<float>();
}