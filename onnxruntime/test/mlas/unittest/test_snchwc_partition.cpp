// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*++

Module Name:

    test_snchwc_partition.cpp

Abstract:

    Tests the cost-to-work-index mapping used by the NCHWc grouped convolution
    algorithms to partition work proportionally to FLOP cost.

    The mapping is exercised directly rather than through a convolution so that
    the two properties the partitioning depends on -- exact tiling of the work
    space, and balanced cost per thread -- can be asserted across shapes that
    would be expensive to convolve.

--*/

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "core/mlas/lib/mlasi.h"

namespace {

//
// Mirrors the derived quantities computed by MLAS_NCHWC_GROUPED_CONV_ALGORITHM
// and PrepareWorkWeighted for a given convolution shape.
//

struct PartitionShape {
  size_t BlockSize;
  size_t FilterSetSize;
  size_t OutputChannels;
  size_t OutputHeight;
  size_t BatchGroupCount;  // BatchCount * GroupCount

  size_t TotalBlockedFilters() const { return OutputChannels / BlockSize; }

  size_t FilterSetCount() const {
    return (OutputChannels + (BlockSize * FilterSetSize) - 1) / (BlockSize * FilterSetSize);
  }

  size_t LastSetFilterCount() const {
    return TotalBlockedFilters() - (FilterSetCount() - 1) * FilterSetSize;
  }

  size_t TotalCost() const { return BatchGroupCount * TotalBlockedFilters() * OutputHeight; }

  size_t TotalWork() const { return BatchGroupCount * FilterSetCount() * OutputHeight; }

  size_t CostToWorkIndex(size_t Cost) const {
    return MlasNchwcCostToWorkIndex(Cost, OutputHeight, FilterSetCount(), FilterSetSize,
                                    TotalBlockedFilters(), LastSetFilterCount());
  }

  //
  // Cost of a single work item: items of the ragged last filter set of each
  // (batch, group) segment cost less than items of full sets.
  //

  size_t WorkItemCost(size_t WorkIndex) const {
    const size_t Set = (WorkIndex / OutputHeight) % FilterSetCount();
    return (Set + 1 == FilterSetCount()) ? LastSetFilterCount() : FilterSetSize;
  }

  size_t RangeCost(size_t Begin, size_t End) const {
    size_t Cost = 0;
    for (size_t i = Begin; i < End; i++) {
      Cost += WorkItemCost(i);
    }
    return Cost;
  }

  std::string Describe() const {
    return "OutputChannels=" + std::to_string(OutputChannels) +
           " OutputHeight=" + std::to_string(OutputHeight) +
           " BatchGroupCount=" + std::to_string(BatchGroupCount) +
           " FilterSetCount=" + std::to_string(FilterSetCount()) +
           " LastSetFilterCount=" + std::to_string(LastSetFilterCount());
  }
};

//
// The shape grid. Output channel counts are chosen to cover every ragged tail
// the mapping has to handle: 16 and 32 are shorter than one filter set, 64 and
// 256 divide evenly, and 48/96/144/208 leave last sets of 3, 2, 1 and 1 blocks.
//

std::vector<PartitionShape> AllShapes() {
  const size_t OutputChannelValues[] = {16, 32, 48, 64, 96, 144, 208, 256};
  const size_t OutputHeightValues[] = {1, 2, 7, 13, 52};
  const size_t BatchGroupValues[] = {1, 2, 8};

  std::vector<PartitionShape> Shapes;

  for (size_t OutputChannels : OutputChannelValues) {
    for (size_t OutputHeight : OutputHeightValues) {
      for (size_t BatchGroupCount : BatchGroupValues) {
        Shapes.push_back(PartitionShape{16, 4, OutputChannels, OutputHeight, BatchGroupCount});
      }
    }
  }

  return Shapes;
}

const ptrdiff_t ThreadCounts[] = {1, 2, 3, 4, 5, 8, 16, 32};

}  // namespace

//
// Partitioning a cost interval and converting both endpoints must produce work
// ranges that exactly tile [0, TotalWork): no gap, no overlap, nothing dropped.
// The convolution's correctness rests entirely on this property -- a gap would
// silently leave output rows uncomputed.
//

TEST(SnchwcPartition, RangesTileWorkSpace) {
  for (const PartitionShape& Shape : AllShapes()) {
    for (ptrdiff_t tids : ThreadCounts) {
      size_t Expected = 0;

      for (ptrdiff_t Index = 0; Index < tids; Index++) {
        size_t CostStart;
        size_t CostCount;
        MlasPartitionWork(Index, tids, Shape.TotalCost(), &CostStart, &CostCount);

        const size_t Begin = Shape.CostToWorkIndex(CostStart);
        const size_t End = Shape.CostToWorkIndex(CostStart + CostCount);

        ASSERT_LE(Begin, End) << Shape.Describe() << " tids=" << tids << " Index=" << Index;
        ASSERT_EQ(Begin, Expected) << "gap or overlap; " << Shape.Describe()
                                   << " tids=" << tids << " Index=" << Index;
        Expected = End;
      }

      ASSERT_EQ(Expected, Shape.TotalWork())
          << "work space not fully covered; " << Shape.Describe() << " tids=" << tids;
    }
  }
}

//
// The mapping must be monotone and bounded over the whole cost domain, which is
// what lets the tiling above hold for any partition MlasPartitionWork produces.
//

TEST(SnchwcPartition, MonotoneAndBounded) {
  for (const PartitionShape& Shape : AllShapes()) {
    size_t Previous = 0;

    for (size_t Cost = 0; Cost <= Shape.TotalCost(); Cost++) {
      const size_t WorkIndex = Shape.CostToWorkIndex(Cost);

      ASSERT_GE(WorkIndex, Previous) << "not monotone at Cost=" << Cost << "; " << Shape.Describe();
      ASSERT_LE(WorkIndex, Shape.TotalWork())
          << "out of bounds at Cost=" << Cost << "; " << Shape.Describe();

      Previous = WorkIndex;
    }

    ASSERT_EQ(Shape.CostToWorkIndex(0), size_t{0}) << Shape.Describe();
    ASSERT_EQ(Shape.CostToWorkIndex(Shape.TotalCost()), Shape.TotalWork()) << Shape.Describe();
  }
}

//
// The point of the weighted split: each thread's share of the actual work --
// measured in block-rows, not work items -- must land near TotalCost/tids.
// A uniform split by work index fails this whenever the last filter set is
// ragged, which is the regression this guards.
//

TEST(SnchwcPartition, CostIsBalancedAcrossThreads) {
  for (const PartitionShape& Shape : AllShapes()) {
    for (ptrdiff_t tids : ThreadCounts) {
      if (Shape.TotalCost() < static_cast<size_t>(tids)) {
        continue;  // covered by EmptyPartitionsAreWellFormed
      }

      const size_t Ideal = Shape.TotalCost() / static_cast<size_t>(tids);

      //
      // A thread's range boundary can only fall on a work item, so its cost can
      // deviate from ideal by at most the cost of one item at each end.
      //

      const size_t Tolerance = 2 * Shape.FilterSetSize;

      for (ptrdiff_t Index = 0; Index < tids; Index++) {
        size_t CostStart;
        size_t CostCount;
        MlasPartitionWork(Index, tids, Shape.TotalCost(), &CostStart, &CostCount);

        const size_t Begin = Shape.CostToWorkIndex(CostStart);
        const size_t End = Shape.CostToWorkIndex(CostStart + CostCount);
        const size_t ActualCost = Shape.RangeCost(Begin, End);

        const size_t Deviation = (ActualCost > Ideal) ? (ActualCost - Ideal) : (Ideal - ActualCost);

        ASSERT_LE(Deviation, Tolerance)
            << "thread " << Index << " of " << tids << " got " << ActualCost
            << " block-rows, ideal " << Ideal << "; " << Shape.Describe();
      }
    }
  }
}

//
// When there is less cost than there are threads, the surplus threads receive an
// empty interval starting at TotalCost. That must map to TotalWork so the caller
// computes WorkRemaining == 0 and skips SeekToWork -- seeking to TotalWork would
// advance the convolution buffer pointers past the end of their tensors.
//

TEST(SnchwcPartition, EmptyPartitionsAreWellFormed) {
  for (const PartitionShape& Shape : AllShapes()) {
    for (ptrdiff_t tids : ThreadCounts) {
      if (Shape.TotalCost() >= static_cast<size_t>(tids)) {
        continue;
      }

      bool SawEmpty = false;

      for (ptrdiff_t Index = 0; Index < tids; Index++) {
        size_t CostStart;
        size_t CostCount;
        MlasPartitionWork(Index, tids, Shape.TotalCost(), &CostStart, &CostCount);

        const size_t Begin = Shape.CostToWorkIndex(CostStart);
        const size_t End = Shape.CostToWorkIndex(CostStart + CostCount);

        if (CostCount == 0) {
          SawEmpty = true;
          ASSERT_EQ(Begin, End) << "empty cost interval produced work; " << Shape.Describe()
                                << " tids=" << tids << " Index=" << Index;
          ASSERT_LE(Begin, Shape.TotalWork())
              << "empty partition seeks out of bounds; " << Shape.Describe()
              << " tids=" << tids << " Index=" << Index;
        }
      }

      ASSERT_TRUE(SawEmpty) << "expected surplus threads; " << Shape.Describe()
                            << " tids=" << tids;
    }
  }
}

//
// The worked example from the mapping's documentation: 96 output channels give
// 6 blocked filters, so two filter sets costing 4 and 2 block-rows per output
// row. With 4 threads a uniform split by work index would hand two threads the
// 4-block set and two the 2-block set -- a 2x imbalance -- while the weighted
// split moves the boundary into the cheaper set to even the cost out.
//

TEST(SnchwcPartition, WorkedExampleFromDocumentation) {
  const PartitionShape Shape{16, 4, 96, 52, 1};

  ASSERT_EQ(Shape.TotalBlockedFilters(), size_t{6});
  ASSERT_EQ(Shape.FilterSetCount(), size_t{2});
  ASSERT_EQ(Shape.LastSetFilterCount(), size_t{2});
  ASSERT_EQ(Shape.TotalCost(), size_t{6 * 52});
  ASSERT_EQ(Shape.TotalWork(), size_t{2 * 52});

  const ptrdiff_t tids = 4;
  const size_t Ideal = Shape.TotalCost() / tids;  // 78 block-rows

  size_t Worst = 0;

  for (ptrdiff_t Index = 0; Index < tids; Index++) {
    size_t CostStart;
    size_t CostCount;
    MlasPartitionWork(Index, tids, Shape.TotalCost(), &CostStart, &CostCount);

    const size_t ActualCost = Shape.RangeCost(Shape.CostToWorkIndex(CostStart),
                                              Shape.CostToWorkIndex(CostStart + CostCount));
    Worst = std::max(Worst, ActualCost);
  }

  //
  // Scheduling efficiency is Ideal/Worst. The uniform split gives 4*52/(2*52*... )
  // -- concretely, its worst thread takes 104 block-rows against an ideal of 78,
  // i.e. 75%. The weighted split must stay within one work item of ideal.
  //

  ASSERT_LE(Worst, Ideal + Shape.FilterSetSize)
      << "worst thread took " << Worst << " block-rows against ideal " << Ideal;
}
