#pragma once

#include <gsl/gsl>

#include <array>
#include <numeric>

namespace ss
{
	/*
		N-dimensional span of `T`
	 */
	template <typename T, size_t N>
	struct ndspan
	{
		explicit ndspan()
			: span{ nullptr, 0 }, shape()
		{
			static_assert(N == 0, "An empty ndspan must be dimensionless");
		}

		ndspan(gsl::span<T> s)
			: span{ s }, shape{ s.size() }
		{
			static_assert(N == 1, "Shape must be specified");
		}

		ndspan(gsl::span<T> s, std::array<ptrdiff_t, N> l)
			: span{ s }, shape{ l }
		{
			Ensures(std::accumulate(shape.begin(), shape.end(), ptrdiff_t{ 1u }, std::multiplies<ptrdiff_t>())
				== span.size());
		}

		gsl::span<T> span;
		const std::array<ptrdiff_t, N> shape;
	};
}