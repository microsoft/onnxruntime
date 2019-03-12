// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "boost/program_options.hpp"

#include "beast_http.h"
#include "core/session/inference_session.h"

namespace po = boost::program_options;
namespace beast = boost::beast;
namespace http = beast::http;

void test_request(const std::string& name, const std::string& version,
                  const std::string& action, onnxruntime::hosting::HttpContext& context) {
  std::stringstream ss;

  ss << "\tModel Name: " << name << std::endl;
  ss << "\tModel Version: " << version << std::endl;
  ss << "\tAction: " << action << std::endl;
  ss << "\tHTTP method: " << context.request.method() << std::endl;

  http::response<http::string_body>
      res{std::piecewise_construct, std::make_tuple(ss.str()), std::make_tuple(http::status::ok, context.request.version())};

  res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
  res.set(http::field::content_type, "plain/text");
  res.keep_alive(context.request.keep_alive());
  context.response = res;
}

int main(int argc, char* argv[]) {
  // TODO: create configuration class for all config related params
  std::string model_path;
  std::string address;
  int port;
  int threads;

  po::options_description desc("Allowed options");
  desc.add_options()
  ("help,h", "Print a help message")
  ("address,a", po::value(&address), "The base HTTP address")
  ("port,p", po::value(&port), "HTTP port to listen to requests")
  ("threads,t", po::value(&threads), "Number of http threads")
  ("model_path,m", po::value(&model_path), "Path of the model file");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);  // can throw

    if (vm.count("help")) {
      std::cout << "ONNX Hosting: host an ONNX model for inferencing with ONNXRuntime\n"
                << std::endl
                << desc << std::endl;
      return EXIT_SUCCESS;
    }

    po::notify(vm);  // throws on error, so do after help
  } catch (po::error& e) {
    std::cerr << "An error with program arguments occurred with error: " << e.what() << std::endl
              << std::endl;
    std::cerr << desc << std::endl;
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << "An unknown problem occurred with error: " << e.what() << std::endl
              << std::endl;
    return EXIT_FAILURE;
  }

  onnxruntime::SessionOptions options {};
  onnxruntime::InferenceSession session(options);

  auto const boost_address = boost::asio::ip::make_address(vm["address"].as<std::string>());

  onnxruntime::hosting::App app {};
  app.Post(R"(/v1/models/([^/:]+)(?:/versions/(\d+))?:(classify|regress|predict))", test_request)
     .Bind(boost_address, vm["port"].as<int>())
     .NumThreads(vm["threads"].as<int>())
     .Run();

  return EXIT_SUCCESS;
}
