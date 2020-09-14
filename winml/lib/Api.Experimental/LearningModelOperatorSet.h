#pragma once

#include "LearningModelOperatorSet.g.h"

namespace WINML_EXPERIMENTALP {

    struct LearningModelOperatorSet : LearningModelOperatorSetT<LearningModelOperatorSet>
    {
        LearningModelOperatorSet(winml_experimental::LearningModelBuilder builder);

        winml_experimental::LearningModelBuilder Add(winml_experimental::LearningModelOperator const& op);

       private:
        winml_experimental::LearningModelBuilder builder_;
        wfc::IVector<winml_experimental::LearningModelOperator> operators_;
    };
}
