#define GSL_THROW_ON_CONTRACT_VIOLATION 1

#include "ss/ss.h"

#include <gtest/gtest.h>
#include <array>
#include <vector>

TEST(ndspan, 2d_constructors)
{
	std::array<float, 8> data;
	ss::ndspan<float, 2> mat(data, { 4, 2 });
	ss::ndspan<float, 2> mat2(data, { 8, 1 });

	/* 4 * 1 != 8 */
	EXPECT_ANY_THROW(( ss::ndspan<float, 2>(data, { 4, 1 }) ));
	/* n > 0 */
	EXPECT_ANY_THROW(( ss::ndspan<float, 2>(data, { 8, 0 }) ));
}