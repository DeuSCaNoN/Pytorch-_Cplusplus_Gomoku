#include "pch.h"
#include "Utils.h"

namespace Utils
{
	int ConvertToIndex(short row, short col, int sideLength)
	{
		return (row * sideLength) + col;
	}

	void ConvertToRowCol(
		int index,
		int sideLength,
		short& row, /*out*/
		short& col /*out*/)
	{
		row = index / sideLength;
		col = index % sideLength;
	}
}