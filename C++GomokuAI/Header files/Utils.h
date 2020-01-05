#pragma once

#define ConvertToIndex(row, col, sideLength) (row * sideLength) + col

namespace Utils
{
	void ConvertToRowCol(int index, int sideLength, short& row, short& col);
}
