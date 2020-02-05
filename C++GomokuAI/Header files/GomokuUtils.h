#pragma once

#define ConvertToIndex(row, col, sideLength) (row * sideLength) + col

namespace GomokuUtils
{
	void ConvertToRowCol(int index, int sideLength, short& row, short& col);

	void GenerateDataSet(short boardSide);
}
