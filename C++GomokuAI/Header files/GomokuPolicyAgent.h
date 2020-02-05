#pragma once

#include <vector>
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

class GomokuPolicyAgent
{
public:
	GomokuPolicyAgent();

	~GomokuPolicyAgent();

	void SaveModel();

	void StartTraining(bool bTurn);

	double PredictValue(char* board, int size, int lastMoveIndex, bool bTurn);

	std::vector<double> PredictMove(char* board, int size, int lastMoveIndex, bool bTurn);
private:

	PyObject* CreateNumpyBoard_(char* board, int size);

	PyObject* m_pyAgent;
	PyObject* m_pNumpyModule;
};