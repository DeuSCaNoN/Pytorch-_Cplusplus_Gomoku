#pragma once

namespace Utils
{
	int ConvertToIndex(short row, short col, int sideLength);
	void ConvertToRowCol(int index, int sideLength, short& row, short& col);
}
