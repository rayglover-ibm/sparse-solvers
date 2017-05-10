#include <kernelpp/kernel_invoke.h>

#include "ss/ss.h"

#include "solvers/homotopy.h"

namespace ss
{
    /* Homotopy solver ----------------------------------------------------- */

	struct homotopy::state {};

	homotopy::homotopy() : m{ nullptr } {}
	homotopy::~homotopy()  = default;

	kernelpp::maybe<ss::homotopy_report> homotopy::solve(
		const ndspan<float, 2> A,
		const gsl::span<float> y,
		      float            tolerance,
		      std::uint32_t    max_iterations,
		      gsl::span<float> x)
	{
		return kernelpp::run<solve_homotopy>(
			A.span.data(), A.shape[0], A.shape[1],
			max_iterations,
			tolerance,
			y.data(),
			x.data());
	}
}