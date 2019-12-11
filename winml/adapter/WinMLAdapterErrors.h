#pragma once

#include "core/common/status.h"

inline __declspec(noinline) winrt::hresult_error _winmla_to_hresult() noexcept {
  try {
    throw;
  } catch (winrt::hresult_error const& e) {
    return e;
  } catch (wil::ResultException const& e) {
    return winrt::hresult_error(e.GetErrorCode(), winrt::to_hstring(e.what()));
  } catch (std::bad_alloc const&) {
    return winrt::hresult_error(E_OUTOFMEMORY);
  } catch (std::out_of_range const& e) {
    return winrt::hresult_out_of_bounds(winrt::to_hstring(e.what()));
  } catch (std::invalid_argument const& e) {
    return winrt::hresult_invalid_argument(winrt::to_hstring(e.what()));
  } catch (onnxruntime::OnnxRuntimeException const& e) {
    StatusCode eStatusCode = static_cast<StatusCode>(e.GetStatus().Code()); 
    return winrt::hresult_error(StatusCodeToHRESULT(eStatusCode), winrt::to_hstring(e.GetStatus().ErrorMessage()));
  } catch (std::exception const& e) {
    return winrt::hresult_error(E_FAIL, winrt::to_hstring(e.what()));
  } catch (...) {
    return winrt::hresult_error(E_FAIL);
  }
}

#define WINMLA_CATCH_ALL  \
  catch (...) {          \
    throw _winmla_to_hresult(); \
  }

#define WINMLA_CATCH_ALL_COM        \
  catch (...) {                    \
    return _winmla_to_hresult().to_abi(); \
  }

#define WINMLA_CATCH_ALL_DONOTHING \
  catch (...) {                   \
    return;                       \
  }