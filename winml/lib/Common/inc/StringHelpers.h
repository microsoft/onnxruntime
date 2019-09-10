#pragma once

// String Helpers
namespace Windows::AI::MachineLearning::Strings
{
    struct hstring_builder
    {
        hstring_builder(hstring_builder const&) = delete;
        hstring_builder& operator=(hstring_builder const&) = delete;

        explicit hstring_builder(UINT32 size)
        {
            winrt::check_hresult(WindowsPreallocateStringBuffer(size, &m_data, &m_buffer));
        }

        ~hstring_builder() noexcept
        {
            if (m_buffer != nullptr) {
                WindowsDeleteStringBuffer(m_buffer);
            }
        }

        wchar_t* data() noexcept
        {
            return m_data;
        }

        winrt::hstring to_hstring()
        {
            winrt::hstring result;
            winrt::check_hresult(WindowsPromoteStringBuffer(m_buffer, reinterpret_cast<HSTRING*>(put_abi(result))));
            m_buffer = nullptr;
            return result;
        }

    private:
        wchar_t* m_data { nullptr };
        HSTRING_BUFFER m_buffer { nullptr };
    };

    static winrt::hstring hstring_from_utf8(const char* input, size_t input_length)
    {
        if (input_length == 0) {
            return {};
        } else if (input_length <= (std::numeric_limits<size_t>::max)()) {
            int output_length = MultiByteToWideChar(CP_UTF8, 0, input, static_cast<int>(input_length), nullptr, 0);
            if (output_length > 0) {
                hstring_builder buffer(static_cast<UINT32>(output_length));
                MultiByteToWideChar(CP_UTF8, 0, input, static_cast<int>(input_length), buffer.data(), output_length);
                return buffer.to_hstring();
            } else {
                winrt::throw_hresult(E_INVALIDARG);
            }
        } else {
            winrt::throw_hresult(E_INVALIDARG);
        }
    }

    static winrt::hstring hstring_from_utf8(const char* input)
    {
        return input != nullptr
            ? hstring_from_utf8(input, strlen(input))
            : L"";
    }

    static winrt::hstring hstring_from_utf8(const std::string& input)
    {
        return hstring_from_utf8(input.c_str(), input.size());
    }

    static std::string utf8_from_unicode(const wchar_t* input, size_t input_length)
    {
        if (input_length == 0) {
            return {};
        } else if (input_length <= (std::numeric_limits<size_t>::max)()) {
            int output_length = WideCharToMultiByte(CP_UTF8, 0, input, static_cast<int>(input_length), nullptr, 0, nullptr, nullptr);
            if (output_length > 0) {
                std::string output(output_length, 0);
                WideCharToMultiByte(CP_UTF8, 0, input, static_cast<int>(input_length), &output[0], output_length, nullptr, nullptr);
                return output;
            } else {
                winrt::throw_hresult(E_INVALIDARG);
            }
        } else {
            winrt::throw_hresult(E_INVALIDARG);
        }
    }

    static std::string utf8_from_hstring(const winrt::hstring& input)
    {
        return utf8_from_unicode(input.data(), input.size());
    }

    static std::wstring wstring_from_string(const std::string& str)
    {
        std::wostringstream woss;
        woss << str.data();
        return woss.str();
    }

} // namespace Windows::AI::MachineLearning::Strings
