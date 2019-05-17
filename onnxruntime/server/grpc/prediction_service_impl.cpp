#include "prediction_service.grpc.pb.h"
#include "../../environment.h"
#include "../../executor.h"
#include <grpc++/grpc++.h>

namespace onnxruntime{
    namespace server {
        namespace grpc{
class PredictionServiceImpl final : public onnxruntime::server::PredictionService::Service {
    private:
    std::shared_ptr<onnxruntime::server::ServerEnvironment>& env;
    
    ::grpc::Status Predict(::grpc::ServerContext* context, const ::onnxruntime::server::PredictRequest* request, ::onnxruntime::server::PredictResponse* response){
        onnxruntime::server::Executor executor(env.get(), "requestid");
        PredictResponse predict_response{};
        auto status = executor.Predict("name", "version", *request, predict_response);
        if (!status.ok()) {
            return ::grpc::Status(::grpc::StatusCode(::grpc::StatusCode::INVALID_ARGUMENT), "error");
        }
        return ::grpc::Status();
    }

};
}
}

}