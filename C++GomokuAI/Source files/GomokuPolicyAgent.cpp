#include "pch.h"

#include "GomokuPolicyAgent.h"

#include <Windows.h>
#include <iostream>
#include <fstream>

#define BOARD_SIDE 15
/*--------------------------------------------------------------*/

GomokuPolicyAgent::GomokuPolicyAgent()
{
	PyObject *pModule, *pDict, *pPythonClass;
	Py_Initialize();

	// Load GomokuAi
	pModule = PyImport_ImportModule("GomokuPolicyAgent");

	if (!pModule)
	{
		std::cerr << "Could not load GomokuPolicyAgnet." << std::endl;
		exit(-1);
	}

	pDict = PyModule_GetDict(pModule);
	if (!pDict) {
		std::cerr << "Failed to get the GomokuPolicyAgent dictionary." << std::endl;
		exit(-1);
	}
	Py_DECREF(pModule);

	pPythonClass = PyDict_GetItemString(pDict, "GomokuPolicyAgent");
	if (!pPythonClass) {
		std::cerr << "Cannot instantiate the GomokuPolicyAgent class." << std::endl;
		exit(-1);
	}
	Py_DECREF(pDict);

	m_pyAgent = PyObject_CallObject(pPythonClass, nullptr);
	Py_DECREF(pPythonClass);

	WIN32_FIND_DATAA fd = { 0 };
	HANDLE hFound = FindFirstFileA("QN15Con5.h5py", &fd);
	bool retval = hFound != INVALID_HANDLE_VALUE;
	FindClose(hFound);
	if (!retval)
	{
		PyObject_CallMethod(m_pyAgent, "CreateModel", "(i)", BOARD_SIDE);
	}
	else
	{
		PyObject_CallMethod(m_pyAgent, "LoadModel", "(s)", "J:\\AITicTacToe\\C++GomokuAI\\x64\\Debug\\QN15Con5.h5py");
	}

	Py_DECREF(pPythonClass);
}

GomokuPolicyAgent::~GomokuPolicyAgent()
{
	Py_DECREF(m_pyAgent);
	if (m_pNumpyModule)
		Py_DECREF(m_pNumpyModule);
	Py_Finalize();
}

/*--------------------------------------------------------------*/

void GomokuPolicyAgent::SaveModel()
{
	PyObject_CallMethod(m_pyAgent, "SaveModel", nullptr);
}

void GomokuPolicyAgent::StartTraining(bool bTurn)
{
	PyObject_CallMethod(m_pyAgent, "StartTraining", "(i)", bTurn ? 1 : 2);
}

double GomokuPolicyAgent::PredictValue(char* board, int size, int lastMoveIndex, bool bTurn)
{
	PyObject* pBoard = CreateNumpyBoard_(board, size);
	PyObject* pValue = PyObject_CallMethod(m_pyAgent, "PredictValue", "(Obi)", pBoard, bTurn, lastMoveIndex);
	double value = PyFloat_AsDouble(pValue);
	Py_DECREF(pBoard);
	Py_DECREF(pValue);

	return value;
}

std::vector<double> GomokuPolicyAgent::PredictMove(char* board, int size, int lastMoveIndex, bool bTurn)
{
	PyObject* pBoard = CreateNumpyBoard_(board, size);
	PyObject* pMoveValues = PyObject_CallMethod(m_pyAgent, "PredictMove", "(Obi)", pBoard, bTurn, lastMoveIndex);

	std::vector<double> data;
	if (PyList_Check(pMoveValues)) {
		for (Py_ssize_t i = 0; i < PyList_Size(pMoveValues); i++) {
			PyObject *value = PyList_GetItem(pMoveValues, i);
			data.push_back(PyFloat_AsDouble(value));
			Py_DECREF(value);
		}
	}
	else {
		throw "Passed PyObject pointer was not a list or tuple!";
	}

	Py_DECREF(pBoard);
	Py_DECREF(pMoveValues);

	return data;
}

PyObject* GomokuPolicyAgent::CreateNumpyBoard_(char* board, int size)
{
	if (!m_pNumpyModule)
		m_pNumpyModule = PyImport_ImportModule("numpy");

	if (!m_pNumpyModule)
	{
		std::cerr << "Could not load GomokuPolicyAgnet." << std::endl;
		exit(-1);
	}

	PyObject* boardList = PyList_New(size);
	if (!boardList)
		throw "Unable to allocate memory for Python list";

	for (int i = 0; i < size; i++) {
		PyObject *num = PyLong_FromSize_t((size_t)board[i]);
		if (!num) {
			Py_DECREF(boardList);
			throw "Unable to allocate memory for Python list";
		}
		PyList_SET_ITEM(boardList, i, num);
		Py_DECREF(num);
	}

	PyObject* pFloatBoard = PyObject_CallMethod(m_pNumpyModule, "array", "(O)", boardList);

	return pFloatBoard;
}