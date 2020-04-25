#include "pch.h"

#include "GomokuPolicyAgent.h"
#include "GomokuUtils.h"

#include "THC/THCCachingHostAllocator.h"
#include <fstream>
#include <future>
#include <exception>

#define BATCH_SIZE 64
#define BATCH_VERBOSE_SIZE 3200 // BATCH_SIZE * 50

/*--------------------------------------------------------------*/

Net::Net()
{
	m_residualNetSize = 5;
	// Construct and register two Linear submodules.
	convN1 = register_module("convN1", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 256, 3).padding(1)));
	batchNorm1 = register_module("batchNorm1", torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(256)));

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
		m_residualBatch1.push_back(register_module(batch1Name, torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(256))));
		m_residualBatch2.push_back(register_module(batch2Name, torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(256))));
		m_residualConv1.push_back(register_module(conv1Name, torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1))));
		m_residualConv2.push_back(register_module(conv2Name, torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1))));
	}

	policyN1 = register_module("policyN1", torch::nn::Conv2d(256, 4, 1));
	batchNorm2 = register_module("batchNorm2", torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(4)));
	policyN2 = register_module("policyN2", torch::nn::Linear(4 * 15 * 15, BOARD_SIDE*BOARD_SIDE));

	valueN1 = register_module("valueN1", torch::nn::Conv2d(256, 2, 1));
	batchNorm3 = register_module("batchNorm3", torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(2)));
	valueN2 = register_module("valueN2", torch::nn::Linear(2 * 15 * 15, 64));
	valueN3 = register_module("valueN3", torch::nn::Linear(64, 1));
}

Net::Net(short resNetSize)
	: m_residualNetSize(resNetSize)
{
	Net();
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
	return torch::log_softmax(policyN2->forward(x.reshape({ x.sizes()[0], 4 * 15 * 15 })), 1);
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
	x = torch::relu(valueN2->forward(x.reshape({ x.sizes()[0], 2 * 15 * 15 })));
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
	policy = torch::log_softmax(policyN2->forward(policy.reshape({ policy.sizes()[0], 4 * 15 * 15 })), 1);

	value = torch::relu(batchNorm3->forward(valueN1->forward(x)));
	value = torch::relu(valueN2->forward(value.reshape({ value.sizes()[0], 2 * 15 * 15 })));
	value = torch::tanh(valueN3->forward(value));
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
	torch::Tensor boardTensor = CreateTensorBoard_(board, size, lastMoveIndex, bTurn);
	torch::Tensor valueTensor = m_pNetworkGpu->forwadValue(boardTensor.reshape({1,4,BOARD_SIDE,BOARD_SIDE}).to(torch::kCUDA));
	return valueTensor[0].item<double>();
}

torch::Tensor GomokuPolicyAgent::PredictMove(char* board, int size, int lastMoveIndex, bool bTurn)
{
	torch::Tensor boardTensor = CreateTensorBoard_(board, size, lastMoveIndex, bTurn).to(torch::kCUDA);
	torch::Tensor policyTensor = m_pNetworkGpu->forwadPolicy(boardTensor.reshape({ 1,4,BOARD_SIDE,BOARD_SIDE })).exp();

	return std::move(policyTensor);
}

void GomokuPolicyAgent::Train(std::vector<TrainingExample>& trainingExamples, double learningRate, short epochs)
{
	m_pNetworkGpu->train();
	torch::optim::Adam adamOptimizer(m_pNetworkGpu->parameters(), learningRate);
	size_t setSize = trainingExamples.size();
	std::future<float> promise;

	bool bRepopulate = trainingExamples.size() > BATCH_SIZE;
	torch::Tensor inputTensor;
	torch::Tensor policyAnswerTensor;
	torch::Tensor valueAnswerTensor;
	for (int currentEpoch = 0; currentEpoch < epochs; ++currentEpoch)
	{
		std::cout << "starting epoch " << currentEpoch << std::endl;
		if (bRepopulate || currentEpoch == 0)
		{
			long long batchSize = setSize >= BATCH_SIZE ? BATCH_SIZE : setSize;
			inputTensor = torch::zeros({ batchSize, 4, BOARD_SIDE, BOARD_SIDE });
			policyAnswerTensor = torch::zeros({ batchSize }, torch::TensorOptions(torch::kLong));
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
					BOARD_SIDE*BOARD_SIDE,
					example.lastMove,
					bTurn);

				valueAnswerTensor[batchIndex] = example.boardValue;
				policyAnswerTensor[batchIndex] = example.moveMade;

				++batchIndex;
				++index;

				if (batchIndex >= BATCH_SIZE || index == setSize)
				{
					if (promise.valid())
					{
						float loss = promise.get();
						if (loss < 3.0)
						{
							learningRate = 0.0002;

						}
						else
						{
							learningRate = 0.002;
						}
						adamOptimizer = torch::optim::Adam(m_pNetworkGpu->parameters(), learningRate);
					}
					bool bVerbose = index % BATCH_VERBOSE_SIZE == 0 || index == setSize;
					promise = std::async(&GomokuPolicyAgent::TrainGpuAsync_, this, adamOptimizer, inputTensor, valueAnswerTensor, policyAnswerTensor, bVerbose);

					batchSize = setSize - index;
					if (batchSize < BATCH_SIZE && batchSize > 0)
					{
						inputTensor = torch::zeros({ batchSize, 4, BOARD_SIDE, BOARD_SIDE });
						policyAnswerTensor = torch::zeros({ batchSize }, torch::TensorOptions(torch::kLong));
						valueAnswerTensor = torch::zeros({ batchSize, 1 });
					}

					batchIndex = 0;
				}
			}
		}
		else
		{
			if (promise.valid())
			{
				float loss = promise.get();
				if (loss < 3.0)
				{
					learningRate = 0.0002;
					
				}
				else
				{
					learningRate = 0.002;
				}
				adamOptimizer = torch::optim::Adam(m_pNetworkGpu->parameters(), learningRate);
			}

			bool bVerbose = true;
			promise = std::async(&GomokuPolicyAgent::TrainGpuAsync_, this, adamOptimizer, inputTensor, valueAnswerTensor, policyAnswerTensor, bVerbose);
		}
	}

	if (promise.valid())
	{
		promise.get();
	}
	m_pNetworkGpu->eval();
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

float GomokuPolicyAgent::TrainGpuAsync_(
	torch::optim::Adam adamOptimizer,
	torch::Tensor inputTensor,
	torch::Tensor valueAnswerTensor,
	torch::Tensor policyAnswerTensor,
	bool bVerbose)
{
	adamOptimizer.zero_grad();
	torch::Tensor policyTensor;
	torch::Tensor valueTensor;

	m_pNetworkGpu->forwardBoth(inputTensor.to(torch::kCUDA),
		policyTensor,
		valueTensor);

	torch::Tensor valueLoss = (valueTensor - valueAnswerTensor.to(torch::kCUDA)).pow(2).mean(); // test new loss equation
	torch::Tensor policyLoss = torch::nll_loss(policyTensor, policyAnswerTensor.to(torch::kCUDA));

	torch::Tensor lossTensor = valueLoss + policyLoss;
	lossTensor.backward();
	adamOptimizer.step();

	if (bVerbose)
	{
		std::cout << lossTensor.data() << std::endl;
	}

	return lossTensor.item<float>();
}